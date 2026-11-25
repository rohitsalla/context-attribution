# TikZ / PGFPlots figure sources

Compile `report_figures.tex` with Tectonic, XeLaTeX or pdfLaTeX after installing TikZ/PGFPlots. With Tectonic:

```sh
tectonic report_figures.tex
```

On Overleaf, set `report_figures.tex` as the main file. This builds three landscape pages: method flow, planned calibration flow and completed-results chart. Edit text/layout in the `.tikz.tex` files and styles in `preamble.tex`. The two `.dat` files hold the exact chart estimates and interval distances; no random or pending values are plotted.

The completed results come from the public context-attribution project, scientific source b11f39e. The new calibration study remains pending. In the repository, `python make_report_figures.py` compiles edited sources without overwriting them; `--regenerate` recreates sources/data from the Python definitions and aggregate JSON.
