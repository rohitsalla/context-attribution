# LaTeX figures for the report and presentation

All final figures are drawn with **TikZ or PGFPlots** and compiled by Tectonic. PNG previews are rendered from the compiled PDFs; SVGs are vector exports of those PDF pages. There are no Matplotlib-generated figures in this set.

- **method_flow:** full-page reference, frozen-note deletion and recomputed selection. Same answer throughout; identical original notes for the two compressed protocols.
- **calibration_flow:** planned development-only effect rescaling followed by fresh evaluation. This is a method diagram, not a completed calibration result.
- **completed_results:** the six verified four-reader-average comparisons at 512 tokens, with paired question-bootstrap intervals. Percentages are attribution-error reductions, not answer-accuracy improvements.

## Files

- [Standalone LaTeX source bundle](../../output/pdf/report_figures_latex.zip)
- [Three-page PDF figure pack](../../output/pdf/report_figures.pdf)
- [Individual method PDF](../../output/pdf/method_flow.pdf), [calibration PDF](../../output/pdf/calibration_flow.pdf), [results PDF](../../output/pdf/completed_results.pdf)
- [LaTeX master](latex/report_figures.tex), [shared styles](latex/preamble.tex)
- Editable components: [method TikZ](latex/method_flow.tikz.tex), [calibration TikZ](latex/calibration_flow.tikz.tex), [PGFPlots chart](latex/completed_results.tikz.tex)
- Exact plotted data: [BM25](latex/bm25_results.dat), [RECOMP](latex/recomp_results.dat), generated from the published aggregate JSON.

For a LaTeX paper, include a vector figure with:

```latex
\usepackage{graphicx}
% In the document:
\includegraphics[width=\linewidth]{output/pdf/method_flow.pdf}
```

Use PNG or SVG for slides. Edit labels/layout in the TikZ sources; the default rebuild preserves these edits. `method_flow.mmd` remains an optional conceptual Mermaid version; the PDF/PNG/SVG exports come from LaTeX.

## Rebuild

From the repository root:

```sh
python make_report_figures.py
```

Requirements: Tectonic, Poppler (`pdftoppm`), pypdf and PyMuPDF. The builder compiles the existing LaTeX master, checks its three-page count, extracts individual vector PDFs and renders previews. Use `python make_report_figures.py --regenerate` only when you want to overwrite the TeX/data files from the Python definitions and aggregate JSON. TeX may download required packages/fonts on first use. Temporary logs stay under ignored `tmp/pdfs/`; deliverables are under `output/pdf/`. The chart reads `research/v4/GENERALIZATION_RESULTS.json`; pending new-study results are not inferred.
