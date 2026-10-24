# README visual sources

- `figure_02_relay_framework.png` is a direct rendering of
  `plotresult/figure_02_relay_framework.pdf`, copied without modification from
  the manuscript's `Diagram/fig_split_depth_relay_generation.pdf` (Figure 2).
- `figure_03_throughput_comparison.png` is a direct rendering of the original
  `plotresult/figure_03_throughput_comparison.pdf`. No values, axis ranges,
  labels, or legends were changed. The original nonzero axes are retained.

Regenerate the PNGs with `python docs/render_readme_assets.py` after installing
PyMuPDF. No manual edits are applied to the rendered PNGs. The intended display
is the full width of the GitHub README.

The README uses the existing paper figures as evidence, not a new benchmark
run. The throughput summary describes only the six comparisons in Figure 3.
The other paper figures remain available as PDF links.

The relay deployment table in the root README reproduces the All-cloud,
RelayTree (s=1), and Full-edge rows of Table II from the manuscript. The s=2
row is omitted for a concise presentation, as stated next to the table.
All six reported metrics are retained. The 99.5% uplink reduction is rounded
from `(1 - 16.81 / 3360.40) * 100`; the approximately one-third reduction in
cloud SLM forward passes follows from `1 - 0.7006 / 1.0508`. Bold identifies the
RelayTree row and does not rank the values.
