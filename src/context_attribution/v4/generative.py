"""Stronger 7B compressor control, on 32 development questions per answerer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from ..v2.hf import HFModel
from ..v2.run import raw_reference,evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--model-key',choices=['qwen7','llama8','mistral7'],required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Fresh output required')
    args.output.mkdir(parents=True,exist_ok=True)
    registry=json.loads((args.root/'research/v2/models.json').read_text());paths=json.loads(args.paths.read_text())
    def load(key):
        m=HFModel(paths[key],registry[key]['revision'],model_id=registry[key]['model_id'])
        m.model.generation_config.repetition_penalty=1.;return m
    answerer=load(args.model_key);compressor=answerer if args.model_key=='qwen7' else load('qwen7')
    dataset=args.root/'data/v2-hotpot/dev.jsonl';items=[json.loads(s) for s in dataset.read_text().splitlines()][:32]
    if any(i['split']!='dev' for i in items):raise ValueError('Development only')
    (args.output/'manifest.json').write_text(json.dumps({'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'source_sha256':{str(f.relative_to(args.root)):hashlib.sha256(f.read_bytes()).hexdigest()
                       for folder in ['v2','v4'] for f in (args.root/'src/context_attribution'/folder).glob('*.py')},
        'runtime':{'answerer':answerer.runtime(),'compressor':compressor.runtime()},'budget':512,'n':32,
        'generation':{'greedy':True,'repetition_penalty':1.},'scope':'development only'},indent=2))
    with (args.output/'traces.jsonl').open('w') as handle:
        for item in items:
            raw=raw_reference(item,answerer)
            for row in [raw,evaluate(item,answerer,compressor,'global',512,raw,2)]:
                handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
            print(item['id'],'complete',flush=True)
    (args.output/'completed.json').write_text(json.dumps({'complete':True,'conditions':len(items)*2}))


if __name__=='__main__':main()
