"""Render the README image directly from the original paper PDF.

Requires PyMuPDF: python -m pip install pymupdf
Run from any directory: python docs/render_readme_assets.py
"""

from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def render_first_page(source: Path, destination: Path, scale: float) -> None:
    with fitz.open(source) as document:
        document[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(destination)


if __name__ == "__main__":
    render_first_page(
        ROOT / "plotresult" / "figure_03_throughput_comparison.pdf",
        ASSETS / "figure_03_throughput_comparison.png",
        4,
    )
