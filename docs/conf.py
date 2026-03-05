import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

sys.path.insert(0, os.path.abspath(".."))

project = "sb3-soft"
author = "Mikihisa Yuasa"

try:
    release = pkg_version("sb3-soft")
except PackageNotFoundError:
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "sb3-soft documentation"
html_static_path = ["_static"]
