"""Freeze a second exploratory benchmark without model-based selection."""
import argparse
import hashlib
import json
from pathlib import Path

REVISION='c8f4f8c9465fb69d31a8eae894c3fd509c4ca321'


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/v4-musique'));args=p.parse_args()
    if args.output.exists():raise ValueError('Frozen dataset output already exists')
    raw=[json.loads(s) for s in args.input.read_text().splitlines()];selected=[]
    for hops in [2,3,4]:
        candidates=[r for r in raw if int(r['id'][0])==hops and r['answerable']]
        candidates.sort(key=lambda r:hashlib.sha256(('context-v4-musique|'+r['id']).encode()).hexdigest())
        for r in candidates[:32]:
            sources=[{'id':f'd{i}','title':s['title'],'text':s['paragraph_text']} for i,s in enumerate(r['paragraphs'])]
            selected.append({'id':r['id'],'base_id':r['id'],'question':r['question'],'gold':r['answer'],
                'answer_aliases':r['answer_aliases'],'sources':sources,'split':'dev','hops':hops,
                'support_ids':[f'd{i}' for i,s in enumerate(r['paragraphs']) if s['is_supporting']],
                'support_granularity':'paragraph','dataset':'dgslibisey/MuSiQue','dataset_revision':REVISION,
                'human_reaudited':False})
    args.output.mkdir(parents=True)
    data=args.output/'dev.jsonl';data.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in selected))
    manifest={'dataset':'MuSiQue-Ans v1.0, revision-pinned HF mirror','hf_dataset':'dgslibisey/MuSiQue',
        'revision':REVISION,'official_repository':'https://github.com/StonyBrookNLP/musique','license':'CC-BY-4.0, official repository',
        'selection':'32 answerable questions per 2/3/4-hop stratum, fixed SHA256 ID order; official development split',
        'raw_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'dev_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
        'n':len(selected),'raw_rows':len(raw),'official_zip_byte_equivalence_verified':False,
        'scope':'second exploratory development benchmark; not final confirmation; no model selection',
        'gold_support_use':'evaluation slices and clearly marked oracle comparator only'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
