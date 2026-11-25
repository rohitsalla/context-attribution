"""Render presentation tables from verified aggregate JSON; leave pending cells blank."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parent
MODELS=('qwen7','llama8','mistral7','qwen14')
DATASETS=('hotpot400','musique240','2wiki200')
NAMES={'qwen7':'Qwen2.5-7B','llama8':'Llama3.1-8B','mistral7':'Mistral7B-v0.3','qwen14':'Qwen2.5-14B',
       'hotpot400':'Hotpot (400)','musique240':'MuSiQue (240)','2wiki200':'2Wiki (200)'}
POLICIES=('bm25_frozen','bm25_fixed_rank','bm25_refitted','recomp_frozen','recomp_repacked')
SHORT=('BM25 frozen','BM25 fixed refill','BM25 recomputed','RECOMP frozen','RECOMP repacked')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def main():
    source=ROOT/'research/v4/GENERALIZATION_RESULTS.json';r=json.loads(source.read_text())
    assert len(r['cells'])==12 and all(c['complete'] for c in r['cells'].values())
    pairs=[('BM25','bm25_frozen','bm25_refitted'),('RECOMP','recomp_frozen','recomp_repacked')]
    mean_rows=[];mean_ci_rows=[]
    for dataset in DATASETS:
        for label,left,right in pairs:
            f=mean(r['cells'][model+'/'+dataset]['methods'][left]['singleton']['question_mae'] for model in MODELS)
            d=mean(r['cells'][model+'/'+dataset]['methods'][right]['singleton']['question_mae'] for model in MODELS)
            c=r['dataset_level_model_average'][dataset]['contrasts'][left+'-minus-'+right]
            lo,hi=c['question_bootstrap_95'];assert hi<0
            mean_rows.append([NAMES[dataset],label,f'{f:.4f}',f'{d:.4f}',f'{100*(d-f)/d:.1f}%',f'[{lo:.4f}, {hi:.4f}]'])
            mean_ci_rows.append([NAMES[dataset],label,f'{c["mean"]:.5f}',f'[{lo:.5f}, {hi:.5f}]'])
    reductions=[]
    for model in MODELS:
        row=[NAMES[model]]
        for dataset in DATASETS:
            c=r['cells'][model+'/'+dataset];values=[]
            for _,left,right in pairs:
                f=c['methods'][left]['singleton']['question_mae'];d=c['methods'][right]['singleton']['question_mae']
                uncertain=c['contrasts'][left+'-minus-'+right]['question_bootstrap_95'][1]>=0
                values.append(f'{100*(d-f)/d:.1f}%'+('†' if uncertain else ''))
            row.append(' / '.join(values))
        reductions.append(row)
    status=json.loads((ROOT/'research/v5/STABILITY_STATUS_20261125.json').read_text())
    progress=[]
    for model in ('qwen7','mistral7'):
        for dataset in ('musique','2wiki'):
            c=status['cohorts'][model+'/'+dataset]
            progress.append([NAMES[model],dataset,f"{c['complete_shards']}/8",f"{c['dev_complete']}/{84 if dataset=='musique' else 80}",f"{c['test_complete']}/240",c['status']])
    pending=table(['Fresh cohort, two-reader average','Budget','BM25 calibrated Δ MAE [95% CI]','RECOMP calibrated Δ MAE [95% CI]','Frozen vs zero (BM25 / RECOMP)'],
                  [[dataset,str(budget),'','',''] for dataset in ('MuSiQue','2Wiki') for budget in (256,512,1024,2048)])
    readme='''# Context attribution under compression — presentation README

**Idea:** When auditing a compressed context, deleting a source and rerunning the selector can introduce replacement evidence. We test whether freezing the original selection gives source effects closer to deleting that source from the full context.

For the same raw greedy answer tokens, measure the change in mean answer-token log probability after each source deletion. Compare compressed effects with that raw-context reference using mean absolute error (MAE; lower is better).

- **Frozen deletion:** remove selected chunks belonging to the deleted source; do not refill.
- **Fixed-ranking refill:** keep candidate priorities fixed and repack surviving evidence.
- **Recomputed selection:** refit ranking/repacking after deletion. RECOMP uses independent sentence scores, so its comparison changes packing.

## Experiments run

'''
    readme+=table(['Phase','Scope','Outcome'],[
        ['Replay prototype','120 mixed fictional/facts-page cases; 12 online checks','No established efficiency win over cheap source-ID tracing.'],
        ['Abstractive/source-local pilot','Small development profiles','No demonstrated win over strong lexical selection.'],
        ['Paired / anchored training','Three matched-state seeds; 96 development questions','No reliable special advantage over unpaired supervision.'],
        ['FP32 budget frontier','Qwen/Llama; Hotpot 96 + MuSiQue 32','Larger budgets helped Hotpot; not reliably MuSiQue.'],
        ['Position / replacement controls','Two readers × two benchmarks; 24 questions each','Mechanism unresolved; numerical-equivalence gate failed.'],
        ['Hotpot confirmation + RECOMP follow-up','Same 400 questions; Qwen/Llama','Frozen deletion favored; both refilling and reranking contribute.'],
        ['Four-reader generalization','4 readers × 840 questions; two selectors; 512-token budget','Complete: 3,360 reader–question evaluations; planned criterion passed.'],
        ['Fresh calibration / budget study','644 new questions; 2 readers × 4 budgets','Running: 5,152 configurations; calibration tables pending.']])
    readme+='''

## Completed main results

Four readers: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3 and Qwen2.5-14B. Benchmarks: Hotpot 400, fresh MuSiQue 240 and fresh 2Wiki 200. Use pinned checkpoints, FP32 weights, math attention and fixed raw-answer targets. Hotpot is a reused replication cohort, not fresh confirmation.

**Four-reader averages.** Gain is the reduction in mean singleton attribution MAE. CI is the paired *frozen-minus-recomputed* MAE difference; negative favors frozen. Readers are averaged within question before question bootstrapping.

'''
    readme+=table(['Dataset','Selector','Frozen MAE','Recomputed MAE','Error reduction','Paired 95% CI'],mean_rows)
    readme+='''

All six primary intervals favor frozen deletion. Percentages use the ratio of averaged MAEs, not the average of per-reader percentages.

**Per-reader reductions: BM25 / RECOMP.** These individual intervals are exploratory and unadjusted; † means the RECOMP interval includes zero.

'''
    readme+=table(['Reader','Hotpot','MuSiQue','2Wiki'],reductions)
    readme+='''

**Interpretation:** Frozen deletion improves fidelity to the specified raw-context deletion reference. It does not establish uniquely correct causal credit or improved answer accuracy. On pooled error, frozen BM25 is worse than predicting zero effects in 6/12 cells; RECOMP in 4/12. Source ranking and deletion-answer agreement improve inconsistently. These absolute-utility gaps motivate the new study.

## Current runs — '''+status['as_of']+'''

Both new reader smokes passed. Completed shards passed source/data/runtime, budget and numerical checks (maximum repeated-score discrepancy 0). Planned time-limit requeues have resumed checkpoints successfully; no unresolved execution errors were found. Counts below include **completed shards only**.

'''
    readme+=table(['Reader','Fresh dataset','Complete shards','Dev questions','Test questions','Status'],progress)
    readme+='''

The sixth Qwen MuSiQue shard is running with 18 additional questions saved. No complete fresh cohort is ready yet. The study has 164 development and 480 test questions: 644 unique questions, 1,288 reader–question evaluations and 5,152 reader–question–budget configurations. Repeated readers/budgets are not independent questions.

## Pending calibration / budget results — intentionally empty

Fit a separate nonnegative effect scale per reader/dataset/budget/policy on development questions only, then evaluate untouched test questions. This is **effect rescaling**, not probability calibration. The 512-token contrasts are primary; other budgets are secondary. Blank cells mean unavailable results, never zero or a claimed failure. No partial-cohort fit is reported. Δ means rescaled frozen-minus-recomputed MAE; negative favors frozen.

'''+pending+'''

## Presentation message and next decision

Present a replicated **deletion-protocol effect**. Pairing-specific training, a universal larger-budget solution, and a pure replacement-versus-position mechanism are not established. The next decision is whether the advantage survives development-only rescaling and beats zero prediction in absolute error. Matched-length donor controls are secondary; donors are not neutral padding.

[Full numerical tables, QA, rank metrics and historical controls](research/PRESENTATION_TABLES.md) · [Main aggregate JSON](research/v4/GENERALIZATION_RESULTS.json) · [New frozen protocol](research/v5/STABILITY_PROTOCOL.md) · [New cohort manifest](data/stability/manifest.json) · [Related-work audit](research/v4/LITERATURE_EXPLORATION.md)
'''
    readme+='\nFor a beginner-facing report narrative, diagrams, verified examples and slide order, use [FINAL_REPORT_DESIGN.md](FINAL_REPORT_DESIGN.md).\n'
    (ROOT/'PRESENTATION_README.md').write_text(readme)
    appendix=['# Full presentation tables','', 'These tables reproduce the verified main-study aggregate JSON. No pending calibration values are inferred. Main source: b11f39e; repaired CPU review 8951475. The new study remains source b74241d. Numeric precision is rounded for display; use the [aggregate JSON](v4/GENERALIZATION_RESULTS.json) for full precision.','',
      '## 1. Primary four-reader paired contrasts','',table(['Dataset','Selector','Frozen-minus-recomputed MAE','95% question-bootstrap CI'],mean_ci_rows),'',
      'All six intervals favor frozen deletion. They average fixed readers within each shared question, then resample questions. This is a statement about these readers/datasets, not a population of architectures. Individual reader contrasts below are exploratory/unadjusted.']
    for group,group_name in [('singleton','Singleton source deletion'),('joint','Two predetermined source-pair deletions per question')]:
        for metric,label,percent in [('question_mae','Question-average MAE',False),('pooled_nmae','Pooled normalized MAE',False),('top_effect_hit_rate','Top-effect recovery (%)',True)]:
            rows=[]
            for model in MODELS:
                for dataset in DATASETS:
                    c=r['cells'][model+'/'+dataset];values=[]
                    for policy in POLICIES:
                        value=c['methods'][policy][group][metric]
                        values.append('' if value is None else f'{value*100:.2f}' if percent else f'{value:.4f}')
                    rows.append([NAMES[model],NAMES[dataset],*values])
            appendix+=['',f'## {group_name}: {label}','',table(['Reader','Dataset',*SHORT],rows)]
            if metric=='pooled_nmae':appendix+=['', 'The all-zero predictor has pooled NMAE 1 on these nonzero-reference aggregates. Pooled weighting differs from the primary per-question MAE when source counts vary.']
            if metric=='top_effect_hit_rate':appendix+=['','Rank diagnostics use the source report’s eligibility rule; denominators can differ from the full cohort. Rank recovery is not answer accuracy.']
    rows=[]
    for model in MODELS:
        for dataset in DATASETS:
            c=r['cells'][model+'/'+dataset];rows.append([NAMES[model],NAMES[dataset],f"{100*c['raw_qa_f1']:.2f}",*[f"{100*c['qa_f1'][s]:.2f}" for s in ('bm25','recomp')]])
    appendix+=['','## Factual answer F1 (%)','',table(['Reader','Dataset','Raw','BM25','RECOMP'],rows),'',
       'Frozen and recomputed policies share the same factual compressed memory and factual answer by construction. These factual QA values are not separate before-deletion gains for frozen versus recomputed policies.']
    rows=[]
    for model in MODELS:
        for dataset in DATASETS:
            c=r['cells'][model+'/'+dataset];rows.append([NAMES[model],NAMES[dataset],*[f"{c['factual_positions'][s]:.1f}" for s in ('raw','bm25','recomp')]])
    appendix+=['','## Mean factual memory positions','',table(['Reader','Dataset','Raw memory','BM25 memory','RECOMP memory'],rows),'',
       'Compressed memory is capped at 512 answerer-token positions. These values exclude the surrounding question/chat prompt and answer continuation.']
    rows=[];qa=[]
    for model in MODELS:
        for dataset in DATASETS:
            c=r['cells'][model+'/'+dataset]
            rows.append([NAMES[model],NAMES[dataset],*[f"{100*c['severe_source_decision_match_to_raw'][p]:.2f}" for p in POLICIES]])
            qa.append([NAMES[model],NAMES[dataset],*[f"{100*c['qa_on_severe_source_deletion'][p]:.2f}" for p in ('raw',)+POLICIES]])
    appendix+=['','## Deletion-answer agreement with raw (%)','',table(['Reader','Dataset',*SHORT],rows),'',
       'Diagnostic deletion chooses the source with greatest absolute raw singleton effect. This severity-selected diagnostic is not prevalence over arbitrary omissions.','',
       '## Answer F1 after that severity-selected deletion (%)','',table(['Reader','Dataset','Raw',*SHORT],qa)]
    contrast_rows=[]
    for model in MODELS:
        for dataset in DATASETS:
            for name,c in r['cells'][model+'/'+dataset]['contrasts'].items():
                lo,hi=c['question_bootstrap_95'];contrast_rows.append([NAMES[model],NAMES[dataset],name,f"{c['mean']:.5f}",f'[{lo:.5f}, {hi:.5f}]'])
    appendix+=['','## All individual-reader paired contrasts','',table(['Reader','Dataset','First minus second','Mean MAE difference','95% question-bootstrap CI'],contrast_rows)]
    legacy=[('Replay prototype: mixed synthetic/facts-page data','RESULTS.md'),
            ('Earliest source-local pilot: unstable small development samples, sum-log-p effects','research/v3/PILOT_REVIEW.md'),
            ('Anchored training: adaptive development, not final confirmation','research/v4/ANCHORED_RESULTS.md'),
            ('Earlier FP32 frontier: Hotpot 96 / MuSiQue 32 per reader','research/v4/FP32_COMPLETE_RESULTS.md'),
            ('Position/replacement ablation: numerical gate failed','research/v4/REPLACEMENT_LAYOUT_RESULTS.md'),
            ('Shared 400-question Hotpot secondary follow-up','research/v4/HELDOUT_FOLLOWUP_RESULTS.md')]
    for title,relative in legacy:
        lines=(ROOT/relative).read_text().splitlines();tables=[];block=[]
        for line in lines+['']:
            if line.startswith('|'):block.append(line)
            elif block:tables.append('\n'.join(block));block=[]
        link=relative[len('research/'):] if relative.startswith('research/') else '../'+relative
        appendix+=['',f'## Historical: {title}','',f'Source and qualifications: [{Path(relative).name}]({link}). These cohorts/targets differ from the main matrix; do not pool them as independent confirmation.','', '\n\n'.join(tables)]
        if 'numerical gate' in title:appendix+=['','Three of 1,440 deletions exceeded the preset equivalence tolerance. Equal-length donor intervals included zero throughout. These are diagnostic measurements, not a validated causal-mechanism result.']
    appendix+=['','## Pending fresh-cohort results','',pending,'',
      'No complete cohort is available in the November 25 snapshot. The primary and secondary criteria remain unevaluated. Do not populate these blanks using the older development budgets or the completed 512-token generalization results.','',
      '### Pending matched-length replacement controls (512 tokens)','',
      table(['Reader','Fresh dataset','Selector','Actual-minus-donor raw-effect error','95% CI'],[[NAMES[m],d,s,'',''] for m in ('qwen7','mistral7') for d in ('MuSiQue','2Wiki') for s in ('BM25','RECOMP')]),'',
      'The control holds branch length and compact positions fixed while changing newly admitted content. Donor text is not guaranteed irrelevant. Neither these controls nor raw-deletion fidelity establish unique causal source credit.','',
      '## Provenance','',f'Aggregate SHA256: `{hashlib.sha256(source.read_bytes()).hexdigest()}`.',
      'New-run status comes from STABILITY_STATUS_20261125.json; no model scores were fit or selected to fill pending tables.']
    (ROOT/'research/PRESENTATION_TABLES.md').write_text('\n'.join(appendix)+'\n')
    root_readme=ROOT/'README.md';text=root_readme.read_text()
    notice='> **Current presentation summary:** [Short project README](PRESENTATION_README.md) · [Full numerical tables](research/PRESENTATION_TABLES.md) · [Beginner report design](FINAL_REPORT_DESIGN.md). The four-reader generalization study is complete; the fresh calibration/budget study is running.\n\n'
    if notice not in text:
        first,rest=text.split('\n',1);root_readme.write_text(first+'\n\n'+notice+rest.lstrip('\n'))
    print('Created PRESENTATION_README.md and research/PRESENTATION_TABLES.md')


if __name__=='__main__':main()
