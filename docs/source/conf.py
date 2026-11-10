# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

project = 'rbg'
copyright = '2026, Carlos Eduardo Rocha, Cristiane Magarinos Sampaio, Giovani, Lisandra Fagundes, Luana Fagundes, Lucca Moulin'
author = 'Carlos Eduardo Rocha, Cristiane Magarinos Sampaio, Giovani, Lisandra Fagundes, Luana Fagundes, Lucca Moulin'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	"sphinx.ext.autodoc",      
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",     
    "sphinx.ext.viewcode",     
    "sphinx.ext.mathjax",   
]

templates_path = ['_templates']
exclude_patterns = []
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
