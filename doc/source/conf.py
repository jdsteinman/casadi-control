# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'casadi-control'
copyright = '2026, John Steinman'
author = 'John Steinman'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ["custom.css"]

html_title = "CasADi-Control"   # shows in browser title + header
#html_logo = "_static/logo.svg"               # optional; put your logo here


todo_include_todos = True
autosummary_generate = True
autoclass_content = "both"

# Keep signatures readable; parameter details come from docstrings.
autodoc_typehints = "none"

# Avoids pulling in re-exported stuff
autosummary_imported_members = False

# Avoid noisy duplicated member listings for data-heavy classes.
numpydoc_show_class_members = True
numpydoc_class_members_toctree = True

autosummary_ignore_module_all = False

autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": True,
}

html_theme_options = {
    "show_toc_level": 2,
    "navigation_depth": 4,
}

html_sidebars = {
    "index": ["search-button-field"],
    "**": ["search-button-field", "sidebar-nav-bs"]
}

# Keep import-time failures from breaking docs generation in minimal envs.
autodoc_mock_imports = ["casadi"]
