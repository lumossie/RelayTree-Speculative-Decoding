# README visual sources

- `relaytree-overview.svg` is an editable conceptual diagram created for this
  repository. Its stages follow `TreeMCSDSpecSamplingEnv._generate_draft` and
  `step` in `relaytree/sampling/relay_tree.py`, and the prefix replay/suffix
  generation methods in `relaytree/inference/strategies.py`. It is not a figure
  reproduced from the published paper. The tree is illustrative.
- `relaytree-overview.png` is the rendered README version of the SVG.
- `figure_03_throughput_comparison.png` is a direct rendering of the original
  `plotresult/figure_03_throughput_comparison.pdf`. No values, axis ranges,
  labels, or legends were changed. The original nonzero axes are retained.

Regenerate both PNGs with `python docs/render_readme_assets.py` after installing
PyMuPDF. The SVG contains the editable layout; no manual edits are applied to
the rendered PNGs. The intended display is the full width of the GitHub README.

The README uses the existing paper figures as evidence, not a new benchmark
run. The throughput summary describes only the six comparisons in Figure 3.
The other paper figures remain available as PDF links.
