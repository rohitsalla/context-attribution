"""Resolve pinned public models through a private, site-local path registry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROFILES=[('qwen7','qwen15'),('llama8','qwen15'),('mistral7','qwen15'),('qwen7','qwen7')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--task',type=int,default=0);p.add_argument('--preflight',action='store_true')
    p.add_argument('--check-only',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    registry=json.loads((root/'research/v2/models.json').read_text())
    paths=json.loads(args.paths.read_text())
    for name in registry:
        path=Path(paths[name]);index=path/'model.safetensors.index.json'
        files=list(json.loads(index.read_text())['weight_map'].values()) if index.exists() else ['model.safetensors']
        for filename in set(files)|{'config.json','tokenizer_config.json'}:
            if not (path/filename).is_file():raise FileNotFoundError(f'{name}: missing {filename}')
    if args.check_only:print('All four pinned runtime checkpoints present');return
    if args.task<0 or args.task>5:raise ValueError('Unknown task')
    out=root/'results/v2-dev'
    if args.task<4:
        answer,compressor=PROFILES[args.task]
        cmd=[sys.executable,'-u','-m','context_attribution.v2.run',
             '--dataset',str(root/'data/v2-hotpot/dev.jsonl'),
             '--model',paths[answer],'--model-id',registry[answer]['model_id'],
             '--revision',registry[answer]['revision'],'--compressor',paths[compressor],
             '--compressor-id',registry[compressor]['model_id'],'--compressor-revision',registry[compressor]['revision'],
             '--output',str(out/('preflight' if args.preflight else f'profile{args.task}'))]
        if args.preflight:cmd+=['--limit','1','--budgets','256','--methods','global','source','lead']
    else:
        cmd=[sys.executable,'-u','-m','context_attribution.v2.train_canary',
             '--dataset',str(root/'data/v2-hotpot/train.jsonl'),
             '--model',paths['qwen7'],'--revision',registry['qwen7']['revision'],
             '--encoder',paths['qwen15'],'--encoder-revision',registry['qwen15']['revision'],
             '--lambda-cf',str(args.task-4),'--output',str(out/f'train_cf{args.task-4}')]
    print('Launching',args.task,'preflight',args.preflight,flush=True)
    subprocess.run(cmd,cwd=root,check=True,env=os.environ.copy())


if __name__=='__main__':main()
