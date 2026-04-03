# docs/source/conf.py  (or wherever your conf.py lives)

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'casadi-control'
copyright = '2026, John Steinman'
author = 'John Steinman'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'numpydoc',
    'sphinx_design',
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
