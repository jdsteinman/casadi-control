"""IPOPT solver adapter for CasADi NLP objects."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import casadi as ca

from .._array_utils import as_sized_1d_float_vector
from ..discretization.base import Guess, NLPLike, DiscreteSolution
from .options import IpoptOptions

Array = np.ndarray


# =============================================================================
# Defaults
# =============================================================================

# Canonical CasADi nlpsol options: top-level CasADi + nested plugin dict for IPOPT.
DEFAULT_CASADI_OPTS: Dict[str, Any] = {
    "print_time": False,
}

DEFAULT_IPOPT_OPTS: Dict[str, Any] = {
    # Output / iteration
    "print_level": 5,
    "max_iter": 3000,

    # Barrier / globalization
    "mu_strategy": "adaptive",

    # Linear solver
    "linear_solver": "ma57",

    # Stopping tolerances
    "tol": 1e-8,
    "constr_viol_tol": 1e-6,
    "dual_inf_tol": 1e-6,
    "compl_inf_tol": 1e-6,

    # "Good enough" early exit
    "acceptable_tol": 1e-6,
    "acceptable_constr_viol_tol": 1e-6,

    # Bounds / slacks:
    "bound_relax_factor": 0.0,
#    "bound_push": 1e-2,
#    "bound_frac": 1e-2,
#    "slack_bound_push": 1e-8,
#    "slack_bound_frac": 1e-8,
}
def _status_from_stats(stats: Dict[str, Any]) -> str:
    """Extract a stable status string from CasADi/IPOPT stats."""
    if not stats:
        return "unknown"

#    rs = stats.get("return_status", None)
#    if isinstance(rs, str) and rs:
#        return rs

    succ = stats.get("success", None)
    if isinstance(succ, (bool, np.bool_)):
        return "success" if bool(succ) else "failed"

    return "unknown"


# =============================================================================
# Option plumbing
# =============================================================================

def _split_legacy_options(options: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Backward-compatibility helper.

    Accept legacy styles and split them into:
      casadi_opts (top-level nlpsol opts)
      ipopt_opts  (plugin opts under 'ipopt')

    Supported legacy styles:
      - {"ipopt": {...}}                     (canonical)
      - {"ipopt.max_iter": 1000, ...}        (dotted)
      - mixed with top-level CasADi keys like "print_time"
    """
    casadi_out: Dict[str, Any] = {}
    ipopt_out: Dict[str, Any] = {}

    for k, v in (options or {}).items():
        if k == "ipopt" and isinstance(v, dict):
            ipopt_out.update(v)
        elif isinstance(k, str) and k.startswith("ipopt."):
            ipopt_out[k[len("ipopt."):]] = v
        else:
            casadi_out[k] = v

    return casadi_out, ipopt_out


def _merge_opts(
    *,
    casadi_opts: Optional[Dict[str, Any]] = None,
    ipopt_opts: Optional[Dict[str, Any]] = None,
    legacy_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a canonical nlpsol options dict:
      opts = {**DEFAULT_CASADI_OPTS, **casadi_opts, "ipopt": {**DEFAULT_IPOPT_OPTS, **ipopt_opts}}

    Precedence (last wins):
      defaults < legacy_options < explicit casadi_opts/ipopt_opts

    Rationale: if callers pass both legacy and explicit, explicit should win.
    """
    cas_legacy: Dict[str, Any] = {}
    ipo_legacy: Dict[str, Any] = {}
    if legacy_options:
        cas_legacy, ipo_legacy = _split_legacy_options(legacy_options)

    cas = dict(DEFAULT_CASADI_OPTS)
    cas.update(cas_legacy)
    if casadi_opts:
        cas.update(casadi_opts)

    ipo = dict(DEFAULT_IPOPT_OPTS)
    ipo.update(ipo_legacy)
    if ipopt_opts:
        ipo.update(ipopt_opts)

    cas["ipopt"] = ipo
    return cas


# =============================================================================
# IPOPT solve on an NLP
# =============================================================================

def solve_ipopt(
    nlp: NLPLike,
    *,
    guess: Optional[Guess] = None,
    opts: Optional[IpoptOptions] = None,
    casadi_opts: Optional[Dict[str, Any]] = None,
    ipopt_opts: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,  # backward compat (legacy mixed dict)
    solver_name: str = "ipopt",
) -> DiscreteSolution:
    """
    Solve an NLP using IPOPT via CasADi.

    Parameters
    ----------
    nlp:
        NLP container with prob/lbx/ubx/lbg/ubg/w0.
    guess:
        Optional Guess(w0, mult_x0, mult_g0) to override initialization and warm-start.
    opts:
        Structured solver options object. When provided, values from ``opts``
        are used as defaults and can still be overridden by explicit keyword
        arguments (explicit keyword arguments take precedence).
    casadi_opts:
        CasADi nlpsol options (top-level keys, e.g. print_time).
    ipopt_opts:
        IPOPT options (keys as in IPOPT, e.g. linear_solver, tol, max_iter).
    options:
        Backward-compat mixed dict. Supports:
          - {"ipopt": {...}} or {"ipopt.xxx": ...} for IPOPT
          - other keys treated as CasADi opts

        If both `options` and the split dicts are provided, split dicts take precedence.
    solver_name:
        Name passed to ca.nlpsol(name, "ipopt", ...).

    Returns
    -------
    casadi_control.discretization.base.DiscreteSolution
        Flat primal solution, objective value, optional multipliers, and
        normalized solver statistics payload.
    """
    if opts is not None:
        if casadi_opts is None:
            casadi_opts = dict(opts.casadi)
        if ipopt_opts is None:
            ipopt_opts = dict(opts.ipopt)
        if options is None:
            options = None if opts.legacy is None else dict(opts.legacy)
        if solver_name == "ipopt":
            solver_name = opts.solver_name

    # --- choose initialization
    if guess is None:
        w0 = np.asarray(nlp.w0, float).reshape(-1)
        mult_x0 = None
        mult_g0 = None
        guess_info: Dict[str, Any] = {}
    else:
        w0 = np.asarray(guess.w0, float).reshape(-1)
        mult_x0 = guess.mult_x0
        mult_g0 = guess.mult_g0
        guess_info = dict(guess.info or {})

    w0 = as_sized_1d_float_vector(w0, int(nlp.n_w), name="w0")

    # --- build solver options
    opts = _merge_opts(casadi_opts=casadi_opts, ipopt_opts=ipopt_opts, legacy_options=options)

    # Warm-start encouragement lives under IPOPT options
    if mult_x0 is not None or mult_g0 is not None:
        ipo = opts.setdefault("ipopt", {})
        ipo.setdefault("warm_start_init_point", "yes")
        ipo.setdefault("mu_init", 1e-3)
        ipo.setdefault("bound_push", 1e-8)
        ipo.setdefault("bound_frac", 1e-8)
        ipo.setdefault("slack_bound_push", 1e-8)
        ipo.setdefault("slack_bound_frac", 1e-8)

    solver = ca.nlpsol(solver_name, "ipopt", nlp.prob, opts)

    # --- solver call arguments
    arg: Dict[str, Any] = {
        "x0": w0,
        "lbx": np.asarray(nlp.lbx, float).reshape(-1),
        "ubx": np.asarray(nlp.ubx, float).reshape(-1),
        "lbg": np.asarray(nlp.lbg, float).reshape(-1),
        "ubg": np.asarray(nlp.ubg, float).reshape(-1),
    }

    if mult_x0 is not None:
        arg["lam_x0"] = as_sized_1d_float_vector(mult_x0, int(nlp.n_w), name="lam_x0")
    if mult_g0 is not None:
        arg["lam_g0"] = as_sized_1d_float_vector(mult_g0, int(nlp.n_g), name="lam_g0")

    # --- solve
    res = solver(**arg)

    # --- unpack
    w_opt = np.asarray(res["x"], float).reshape(-1)
    f_opt = float(np.asarray(res["f"], float).reshape(-1)[0])
    mult_x = None
    mult_g = None

    if "lam_x" in res:
        mult_x = np.asarray(res["lam_x"], float).reshape(-1)
    if "lam_g" in res:
        mult_g = np.asarray(res["lam_g"], float).reshape(-1)

    stats = solver.stats() if hasattr(solver, "stats") else {}
    status = _status_from_stats(stats)

    out_stats: Dict[str, Any] = {
        "solver": "ipopt",
        "status": status,
        "casadi_stats": stats,
        "guess": guess_info,
        "casadi_opts": {k: v for k, v in opts.items() if k != "ipopt"},
        "ipopt_opts": dict(opts.get("ipopt", {})),
        "legacy_options_used": options is not None,
    }

    return DiscreteSolution(
        w_opt=w_opt,
        f_opt=f_opt,
        mult_x=mult_x,
        mult_g=mult_g,
        status=status,
        stats=out_stats,
    )
