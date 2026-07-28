# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

project = "Spatial-IQ"
copyright = "2026, NVIDIA Corporation"
author = "NVIDIA Corporation"
release = "0.1.0"

extensions = [
    "myst_parser",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "html_image",
    "attrs_block",
    "attrs_inline",
]

html_theme = "nvidia_sphinx_theme"
html_title = "Spatial-IQ"
html_static_path = ["_static"]
html_css_files = ["spatial_iq.css"]

html_theme_options = {
    "show_prev_next": False,
    # Drop the right-hand "On this page" panel; it's moved into the left sidebar.
    "secondary_sidebar_items": [],
}

# Show the left navigation followed by the page "On this page" TOC underneath.
html_sidebars = {
    "**": ["sidebar-nav-bs", "page-toc"],
}
