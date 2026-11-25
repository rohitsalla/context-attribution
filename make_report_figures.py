"""Build actual TikZ/PGFPlots figures, compile a PDF pack, and render previews."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
from statistics import mean

ROOT=Path(__file__).resolve().parent
TEX=ROOT/'figures/report/latex'
BUILD=ROOT/'tmp/pdfs/report_figures'
PDF=ROOT/'output/pdf'
FIGURES=ROOT/'figures/report'

PREAMBLE=r'''\documentclass[12pt]{article}
\usepackage[paperwidth=15in,paperheight=8.5in,margin=0.35in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\definecolor{ink}{HTML}{19324D}
\definecolor{refblue}{HTML}{4878B8}
\definecolor{frozengreen}{HTML}{31845C}
\definecolor{reorange}{HTML}{C67532}
\definecolor{muted}{HTML}{546779}
\tikzset{
  title/.style={anchor=north west,text=ink,font=\fontsize{25}{29}\selectfont\bfseries,align=left,text width=35.2cm},
  subtitle/.style={anchor=north west,text=muted,font=\fontsize{13}{17}\selectfont,align=left,text width=35.2cm},
  box/.style={draw,rounded corners=5pt,line width=1.1pt,align=center,minimum height=2.6cm,inner xsep=0.3cm,inner ysep=0.2cm,text=ink,font=\fontsize{15}{20}\selectfont},
  before/.style={box,text width=7.0cm},
  after/.style={box,text width=10.1cm},
  outcome/.style={box,text width=7.3cm},
  flow/.style={-{Stealth[length=4mm,width=3mm]},line width=1.3pt},
  rowlabel/.style={anchor=west,font=\fontsize{12}{14}\selectfont\bfseries},
  collabel/.style={font=\fontsize{11}{13}\selectfont\bfseries,text=muted},
  footer/.style={anchor=north west,text=ink,font=\fontsize{12}{16}\selectfont,align=left,text width=35.2cm}
}
'''

METHOD=r'''\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,0) rectangle (35.6,18.3);
\node[title] at (0.1,17.9) {One source removed. Three ways to measure what changed.};
\node[subtitle] at (0.1,16.5) {Keep the same answer fixed in every row. Both note-based protocols start from identical original notes.};
\node[collabel] at (4.0,14.9) {BEFORE DELETION};
\node[collabel] at (17.1,14.9) {AFTER DELETING SOURCE A};
\node[collabel] at (30.1,14.9) {INFLUENCE MEASUREMENT};
\node[rowlabel,text=refblue] at (0.2,14.0) {FULL PAGES};
\node[before,draw=refblue,fill=refblue!6] (rb) at (4.0,12.2) {All source pages};
\node[after,draw=refblue,fill=refblue!6] (ra) at (17.1,12.2) {Remove source A\\Keep all other pages};
\node[outcome,draw=refblue,fill=refblue!6] (ro) at (30.1,12.2) {Full-page influence\\our reference};
\draw[flow,refblue] (rb.east)--(ra.west);
\draw[flow,refblue] (ra.east)--(ro.west);
\node[rowlabel,text=frozengreen] at (0.2,9.7) {FROZEN NOTES};
\node[before,draw=frozengreen,fill=frozengreen!6] (fb) at (4.0,7.9) {Original short notes};
\node[after,draw=frozengreen,fill=frozengreen!6] (fa) at (17.1,7.9) {Remove only notes from A\\Keep the remaining notes};
\node[outcome,draw=frozengreen,fill=frozengreen!6] (fo) at (30.1,7.9) {Frozen-deletion\\influence};
\draw[flow,frozengreen] (fb.east)--(fa.west);
\draw[flow,frozengreen] (fa.east)--(fo.west);
\node[rowlabel,text=reorange] at (0.2,5.4) {NOTES CHOSEN AGAIN};
\node[before,draw=reorange,fill=reorange!7] (ub) at (4.0,3.6) {Same original\\short notes};
\node[after,draw=reorange,fill=reorange!7] (ua) at (17.1,3.6) {Remove A from the source pool\\Choose and pack notes again};
\node[outcome,draw=reorange,fill=reorange!7] (uo) at (30.1,3.6) {Recomputed\\influence};
\draw[flow,reorange] (ub.east)--(ua.west);
\draw[flow,reorange] (ua.east)--(uo.west);
\node[footer] at (0.1,1.15) {Compare each notes-based measurement with the full-page reference. A closer match means lower attribution error.};
\end{tikzpicture}
'''

CALIBRATION=r'''\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,0) rectangle (35.6,18.3);
\node[title] at (0.1,17.9) {A fair adjustment: practice first, test later};
\node[subtitle,text=reorange] at (0.1,16.5) {Planned fresh-cohort analysis. Result cells stay empty until cohorts are complete.};
\node[before,draw=refblue,fill=refblue!6] (dev) at (4.0,12.4) {Practice questions\\164 unique questions};
\node[after,draw=refblue,fill=refblue!6,text width=8.8cm] (fit) at (17.1,12.4) {Fit a score scale\\for each method};
\node[outcome,draw=frozengreen,fill=frozengreen!6] (lock) at (30.1,12.4) {Freeze the scales\\before evaluation};
\draw[flow,refblue] (dev.east)--(fit.west);
\draw[flow,refblue] (fit.east)--(lock.west);
\node[before,draw=refblue,fill=refblue!3] (test) at (4.0,5.9) {New test questions\\480 unique questions};
\node[after,draw=refblue,fill=refblue!3,text width=8.8cm] (eval) at (17.1,5.9) {Evaluate both methods\\at four note budgets};
\node[outcome,draw=reorange,fill=reorange!7] (compare) at (30.1,5.9) {Compare errors\\and the zero baseline};
\draw[flow,refblue] (test.east)--(eval.west);
\draw[flow,refblue] (eval.east)--(compare.west);
\draw[flow,frozengreen,rounded corners=5pt] (lock.south)--(30.1,9.1)--(17.1,9.1)--(eval.north);
\node[fill=white,inner sep=5pt,text=frozengreen,font=\fontsize{13}{16}\selectfont] at (23.4,9.1) {Apply fixed scales};
\node[text=muted,font=\fontsize{12}{15}\selectfont] at (17.1,3.6) {256 / 512 / 1,024 / 2,048 tokens};
\node[footer] at (0.1,1.7) {Scales are fitted separately by reader, dataset, budget and policy. The language models are not retrained.};
\end{tikzpicture}
'''


def make_sources():
    TEX.mkdir(parents=True,exist_ok=True);BUILD.mkdir(parents=True,exist_ok=True);PDF.mkdir(parents=True,exist_ok=True)
    (TEX/'preamble.tex').write_text(PREAMBLE)
    (TEX/'method_flow.tikz.tex').write_text(METHOD)
    (TEX/'calibration_flow.tikz.tex').write_text(CALIBRATION)
    report=json.loads((ROOT/'research/v4/GENERALIZATION_RESULTS.json').read_text())
    models=('qwen7','llama8','mistral7','qwen14');data={'bm25':[],'recomp':[]};labels=[]
    y=5
    for dataset in ('hotpot400','musique240','2wiki200'):
        for selector,left,right,color in [('bm25','bm25_frozen','bm25_refitted','refblue'),('recomp','recomp_frozen','recomp_repacked','reorange')]:
            c=report['dataset_level_model_average'][dataset]['contrasts'][left+'-minus-'+right];lo,hi=c['question_bootstrap_95'];value=c['mean']
            assert lo<=value<=hi<0
            f=mean(report['cells'][m+'/'+dataset]['methods'][left]['singleton']['question_mae'] for m in models)
            r=mean(report['cells'][m+'/'+dataset]['methods'][right]['singleton']['question_mae'] for m in models)
            data[selector].append(f'{value:.17g} {y} {value-lo:.17g} {hi-value:.17g}')
            labels.append(r'\node[anchor=west,text='+color+r',font=\large] at (axis cs:0.015,'+str(y)+r') {'+f'{100*(r-f)/r:.1f}'+r'\% less error};')
            display={'hotpot400':'Hotpot','musique240':'MuSiQue','2wiki200':'2Wiki'}[dataset]
            method='Word-based' if selector=='bm25' else 'Learned'
            labels.append(r'\node[anchor=east,text=ink,font=\normalsize] at (axis cs:-0.254,'+str(y)+r') {'+display+' / '+method+r'};')
            y-=1
    for selector,rows in data.items():(TEX/f'{selector}_results.dat').write_text('mean y errlo errhi\n'+'\n'.join(rows)+'\n')
    plot=r'''\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,0) rectangle (35.6,18.3);
\node[title] at (0.1,17.9) {Keeping selected notes fixed reduced influence error on average};
\node[subtitle] at (0.1,16.5) {Completed 512-token study: three question collections, two note selectors, four fixed readers.};
\begin{axis}[
 at={(4.7cm,3.3cm)},anchor=south west,scale only axis,width=28.0cm,height=11.2cm,
 xmin=-0.25,xmax=0.105,ymin=-0.65,ymax=5.65,
 xtick={-0.25,-0.20,-0.15,-0.10,-0.05,0,0.05,0.10},
 ytick={0,1,2,3,4,5},
 yticklabels={2Wiki / Learned,2Wiki / Word-based,MuSiQue / Learned,MuSiQue / Word-based,Hotpot / Learned,Hotpot / Word-based},
 yticklabel style={font=\normalsize,align=right},xticklabel style={font=\normalsize},scaled x ticks=false,xticklabel={\pgfmathprintnumber[fixed,fixed zerofill,precision=2]{\tick}},
 axis y line=none,axis x line*=bottom,xmajorgrids=true,grid style={draw=gray!15},
 xlabel={Error difference: fixed notes minus notes chosen again\\Negative values favor fixed notes},
 xlabel style={font=\normalsize,align=center},clip=false]
\addplot+[only marks,mark=*,mark size=3.2pt,color=refblue,mark options={fill=refblue,draw=refblue},error bars/.cd,x dir=both,x explicit,error bar style={line width=1.1pt},error mark options={rotate=90,mark size=4pt,line width=1pt}]
 table[x=mean,y=y,x error minus=errlo,x error plus=errhi] {bm25_results.dat};
\addplot+[only marks,mark=*,mark size=3.2pt,color=reorange,mark options={fill=reorange,draw=reorange},error bars/.cd,x dir=both,x explicit,error bar style={line width=1.1pt},error mark options={rotate=90,mark size=4pt,line width=1pt}]
 table[x=mean,y=y,x error minus=errlo,x error plus=errhi] {recomp_results.dat};
\draw[muted,dashed,line width=0.8pt] (axis cs:0,-0.65)--(axis cs:0,5.65);
'''+ '\n'.join(labels)+r'''
\end{axis}
\node[footer,font=\fontsize{11}{14}\selectfont] at (0.1,1.25) {95\% question-bootstrap intervals after averaging four fixed readers. Word-based = BM25; learned = extractive RECOMP.\\Percentages describe attribution-error reductions, not increases in answer accuracy.};
\end{tikzpicture}
'''
    (TEX/'completed_results.tikz.tex').write_text(plot)
    master=r'\input{preamble.tex}'+'\n'+r'\begin{document}'+'\n'
    for i,name in enumerate(('method_flow','calibration_flow','completed_results')):
        if i:master+='\n'+r'\newpage'+'\n'
        master+=r'\noindent\input{'+name+r'.tikz.tex}'+'\n'
    master+=r'\end{document}'+'\n';(TEX/'report_figures.tex').write_text(master)


def build(regenerate=False):
    from pypdf import PdfReader,PdfWriter
    import fitz
    for folder in (TEX,BUILD,PDF):folder.mkdir(parents=True,exist_ok=True)
    if regenerate or not (TEX/'report_figures.tex').exists():make_sources()
    engine=shutil.which('tectonic')
    if not engine:raise RuntimeError('Install Tectonic or compile report_figures.tex with a LaTeX engine supporting TikZ/PGFPlots')
    subprocess.run([engine,'--untrusted','--keep-logs','--outdir',str(BUILD),'report_figures.tex'],cwd=TEX,check=True)
    combined=PDF/'report_figures.pdf';shutil.copy2(BUILD/'report_figures.pdf',combined)
    reader=PdfReader(combined);assert len(reader.pages)==3
    for i,name in enumerate(('method_flow','calibration_flow','completed_results')):
        writer=PdfWriter();writer.add_page(reader.pages[i]);target=PDF/f'{name}.pdf'
        with target.open('wb') as handle:writer.write(handle)
        subprocess.run(['pdftoppm','-singlefile','-png','-r','150',str(target),str(FIGURES/name)],check=True)
        with fitz.open(target) as document:(FIGURES/f'{name}.svg').write_text(document[0].get_svg_image(text_as_path=True))
    with zipfile.ZipFile(PDF/'report_figures_latex.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(TEX.iterdir()):
            if file.suffix in ('.tex','.dat','.md'):bundle.write(file,file.name)
        bundle.write(ROOT/'research/v4/GENERALIZATION_RESULTS.json','data/GENERALIZATION_RESULTS.json')
    print('Built three LaTeX figures:',combined)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerate',action='store_true',help='Regenerate editable TeX/data sources from Python definitions and aggregate JSON before compiling')
    build(parser.parse_args().regenerate)
