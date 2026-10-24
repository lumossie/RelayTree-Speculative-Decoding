"""Render README images from the editable SVG and original paper PDF.

Requires PyMuPDF: python -m pip install pymupdf
Run from any directory: python docs/render_readme_assets.py
"""

from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def render_first_page(source: Path, destination: Path, scale: float) -> None:
    with fitz.open(source) as document:
        if document.is_pdf:
            document[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(destination)
        else:
            with fitz.open("pdf", document.convert_to_pdf()) as rendered:
                rendered[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(destination)


if __name__ == "__main__":
    render_first_page(ASSETS / "relaytree-overview.svg", ASSETS / "relaytree-overview.png", 2)
    render_first_page(
        ROOT / "plotresult" / "figure_03_throughput_comparison.pdf",
        ASSETS / "figure_03_throughput_comparison.png",
        4,
    )
