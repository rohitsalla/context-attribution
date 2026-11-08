"""Expand essential cheap controls to all frozen development questions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from ..v2.launch import PROFILES


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--paths',type=Path,required=True);p.add_argument('--profile',type=int,choices=[0,1,2],required=True)
    args=p.parse_args();models=json.loads((args.root/'research/v2/models.json').read_text());paths=json.loads(args.paths.read_text())
    answer,compressor=PROFILES[args.profile]
    subprocess.run([sys.executable,'-u','-m','context_attribution.v2.run',
        '--dataset',str(args.root/'data/v2-hotpot/dev.jsonl'),'--limit','96',
        '--model',paths[answer],'--model-id',models[answer]['model_id'],'--revision',models[answer]['revision'],
        '--compressor',paths[compressor],'--compressor-id',models[compressor]['model_id'],
        '--compressor-revision',models[compressor]['revision'],'--methods','lead','lexical',
        '--budgets','256','512','--output',str(args.root/f'results/v3/cheap_profile{args.profile}')],check=True)


if __name__=='__main__':main()
