"""Tokenizer-only length/schema preflight; no model weights or answer scores."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from context_attribution.stability_study import DATASETS, MODELS, read_cohort, chosen_rows
from context_attribution.v2.hf import HFModel
from context_attribution.v2.memory import Source
from context_attribution.v2.run import raw_memory


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    registry=json.loads((a.project/'research/v4/GENERALIZATION_MODELS.json').read_text());report={}
    for model in MODELS:
        info=registry[model];tokenizer=AutoTokenizer.from_pretrained(info['model_id'],revision=info['revision'],token=False,trust_remote_code=False)
        formatter=object.__new__(HFModel);formatter.tokenizer=tokenizer;report[model]={}
        for name in DATASETS:
            rows,_=read_cohort(a.project,name);lengths=[]
            assert set(r['id'] for r in rows)==set(r['id'] for k in range(8) for r in chosen_rows(rows,k))
            for row in rows:
                sources=[Source(s['id'],s['title'],s['text']) for s in row['sources']]
                ids=raw_memory(tokenizer,sources);lengths.append(len(formatter.qa_ids(row['question'],ids))+64)
                assert row['split'] in ('dev','test') and row['sources']
            assert max(lengths)<32768
            report[model][name]={'n':len(rows),'max_raw_prompt_plus_64':max(lengths),'smoke_ids':[r['id'] for r in chosen_rows(rows,smoke=True)]}
    path=a.project/'research/v5/STABILITY_PREFLIGHT.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
