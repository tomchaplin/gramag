# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import gramag

project = "gramag"
copyright = "2024, Thomas Chaplin"
author = "Thomas Chaplin"
release = gramag.version()
version = release


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
]

autosummary_generate = True
autosummary_imported_members = True

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = ""
html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]
html_static_path = []
