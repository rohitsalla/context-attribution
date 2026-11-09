"""Conventional global extraction at the anchor's exact factual token allowance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from .v2.hf import HFModel
from .v2.memory import Source
from .v2.metrics import qa,effect_comparison
from .v2.run import raw_memory
from .v3.soft_study import interventions
from .v3.report import summarize
from .v4.evidence import global_lexical


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--suite',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Fresh output required')
    if not (args.suite/'completed.json').exists():raise ValueError('Completed reference suite required')
    dataset=args.root/'data/v2-hotpot/dev.jsonl';items=[json.loads(s) for s in dataset.read_text().splitlines()]
    if any(i['split']!='dev' for i in items):raise ValueError('Development only')
    refs=json.loads((args.suite/'raw_reference.json').read_text())
    literal={r['item_id']:r for r in map(json.loads,(args.suite/'lexical32.jsonl').read_text().splitlines())}
    if set(refs)!=set(literal) or set(refs)!={i['id'] for i in items}:raise ValueError('Reference cohort mismatch')
    paths=json.loads(args.paths.read_text());info=json.loads((args.root/'research/v2/models.json').read_text())['qwen7']
    torch.use_deterministic_algorithms(True)
    model=HFModel(paths['qwen7'],info['revision'],model_id=info['model_id']);model.model.requires_grad_(False)
    model.model.generation_config.repetition_penalty=1.
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'runtime':model.runtime(),'attention_backend':'deterministic SDPA math, matches anchor',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helper_sha256':hashlib.sha256((args.root/'src/context_attribution/v4/evidence.py').read_bytes()).hexdigest(),
        'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256((args.suite/'raw_reference.json').read_bytes()).hexdigest(),
        'literal_trace_sha256':hashlib.sha256((args.suite/'lexical32.jsonl').read_bytes()).hexdigest(),
        'python_hash_seed':os.environ.get('PYTHONHASHSEED'),
        'scope':'factual allowance matched per question; global method refills after deletions, local anchor loses removed slots',
        'generation_repetition_penalty':1.}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));values=[]
    with sdpa_kernel(SDPBackend.MATH),(args.output/'dev.jsonl').open('w') as handle:
        for item in items:
            ref=refs[item['id']];target=ref['answer']['tokens'];n=len(target)
            if target!=literal[item['id']]['raw_answer']['tokens']:raise ValueError('Fixed target mismatch')
            sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']]
            budget=literal[item['id']]['memory_positions']
            raw_base=model.score(model.qa_ids(item['question'],raw_memory(model.tokenizer,sources)),target)
            if abs(raw_base-ref['base'])/n>.001:raise ValueError('Raw-reference score drift exceeds tolerance')
            memory=global_lexical(model.tokenizer,sources,item['question'],budget)
            if len(memory)>budget:raise ValueError('Factual budget exceeded')
            prefix=model.qa_ids(item['question'],memory);answer=model.generate(prefix)
            base=model.score(prefix,target);effects={};positions={}
            for omit in interventions(item):
                key=','.join(map(str,omit));kept=[s for k,s in enumerate(sources) if k not in omit]
                changed=global_lexical(model.tokenizer,kept,item['question'],budget)
                if len(changed)>budget:raise ValueError('Intervention budget exceeded')
                effects[key]=(base-model.score(model.qa_ids(item['question'],changed),target))/n;positions[key]=len(changed)
            truth={k:(ref['base']-v)/n for k,v in ref['changed'].items()}
            single={k:v for k,v in effects.items() if ',' not in k};joint={k:v for k,v in effects.items() if ',' in k}
            row={'item_id':item['id'],'method':'global_lexical_matched_allowance','qa':qa(answer['text'],item['gold']),
                'answer':answer,'raw_qa':ref['qa'],'raw_answer':ref['answer'],'memory_positions':len(memory),'budget':budget,
                'intervention_positions':positions,'common_target_tokens':target,'raw_effects':truth,'effects':effects,
                'single_comparison':effect_comparison({k:truth[k] for k in single},single),
                'joint_comparison':effect_comparison({k:truth[k] for k in joint},joint)}
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno());values.append(row)
            print(item['id'],'F1',row['qa']['f1'],flush=True)
    (args.output/'completed.json').write_text(json.dumps({'complete':True,'n':len(values),'metrics':summarize(values)},indent=2))


if __name__=='__main__':main()
