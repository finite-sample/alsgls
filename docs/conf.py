"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

# _snippets holds fragments spliced into pages via {include}; they are not
# standalone documents, so keep them out of the toctree scan.
configure(
    globals(),
    exclude_patterns=["_build", "Thumbs.db", ".DS_Store", "_snippets"],
)
