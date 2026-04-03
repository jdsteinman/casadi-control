import os
import sys
import json
import shutil
sys.path.insert(0, os.path.abspath('../../src'))

project = 'casadi-control'
copyright = '2026, John Steinman'
author = 'John Steinman'

_here = os.path.abspath(os.path.dirname(__file__))
_repo_root = os.path.abspath(os.path.join(_here, '..', '..'))
_jupyter_root = os.path.join(_here, '.jupyter')
_kernel_dir = os.path.join(_jupyter_root, 'kernels', 'python3')
_mpl_config = os.path.join(_here, '.matplotlib')
_docs_examples_dir = os.path.join(_here, 'examples')
os.makedirs(_kernel_dir, exist_ok=True)
os.makedirs(_mpl_config, exist_ok=True)
with open(os.path.join(_kernel_dir, 'kernel.json'), 'w', encoding='utf-8') as f:
    json.dump(
        {
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        f,
    )
os.environ['JUPYTER_PATH'] = _jupyter_root + os.pathsep + os.environ.get('JUPYTER_PATH', '')
os.environ['MPLCONFIGDIR'] = _mpl_config

for name in ('hager_lq_ocp.ipynb', 'hager_hou_rao_ocp.ipynb'):
    shutil.copy2(
        os.path.join(_repo_root, 'examples', name),
        os.path.join(_docs_examples_dir, name),
    )

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'numpydoc',
    'sphinx_design',
    'sphinxcontrib.bibtex',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build']

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_title = 'CasADi-Control'

todo_include_todos = True
autosummary_generate = True
autoclass_content = 'both'
autodoc_typehints = 'none'
autosummary_imported_members = False
numpydoc_show_class_members = True
numpydoc_class_members_toctree = True
autosummary_ignore_module_all = False
nbsphinx_execute = os.environ.get('NBSPHINX_EXECUTE', 'auto')
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png', 'svg'}",
]

autodoc_default_options = {
    'undoc-members': False,
    'show-inheritance': True,
}

html_theme_options = {
    'show_toc_level': 2,
    'navigation_depth': 4,
}

html_sidebars = {
    'index': ['search-button-field'],
    '**': ['search-button-field', 'sidebar-nav-bs'],
}

autodoc_mock_imports = ['casadi']
bibtex_bibfiles = ['references.bib']
