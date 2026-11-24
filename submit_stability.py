"""Idempotent smoke-gated Slurm submission; at most one study GPU at a time."""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import subprocess


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--python-bin',type=Path,required=True);p.add_argument('--recomp-path',type=Path,required=True)
    p.add_argument('--source-commit',required=True);a=p.parse_args();root=a.project.resolve()
    a.paths=a.paths.resolve();a.python_bin=a.python_bin.absolute();a.recomp_path=a.recomp_path.resolve()
    (root/'logs').mkdir(exist_ok=True)
    with (root/'logs/submission.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(a,root)


def execute(a,root):
    os.chdir(root)
    paths=json.loads(a.paths.read_text());assert all((Path(paths[m])/'config.json').is_file() for m in ('qwen7','mistral7'))
    assert a.python_bin.is_file() and (a.recomp_path/'config.json').is_file()
    assert (root/'data/stability/manifest.json').is_file()
    expected_runtime=json.loads((root/'research/v5/STABILITY_RUNTIME.json').read_text())
    probe='import json,sys,torch,transformers;print(json.dumps(dict(python_major_minor=".".join(map(str,sys.version_info[:2])),torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,prefix=sys.prefix)))'
    runtime=json.loads(subprocess.check_output([str(a.python_bin),'-c',probe],text=True))
    for key,value in expected_runtime.items():
        if runtime.get(key)!=value:raise RuntimeError('Interpreter runtime mismatch: '+repr(runtime))
    print('Validated Python runtime',json.dumps(runtime),flush=True)
    os.environ.update(PROJECT_ROOT=str(root),MODEL_PATHS=str(a.paths.resolve()),PYTHON_BIN=str(a.python_bin.absolute()),RECOMP_PATH=str(a.recomp_path.resolve()))
    (root/'logs').mkdir(exist_ok=True);receipt_path=root/'logs/stability_submission.json'
    identity={'source_commit':a.source_commit,'project':str(root),'model_paths':str(a.paths.resolve()),'python_bin':str(a.python_bin.absolute()),'recomp_path':str(a.recomp_path.resolve())}
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text());assert receipt['identity']==identity
        if receipt.get('released'):
            print(json.dumps(receipt,indent=2));return
    else:
        receipt={'identity':identity,'jobs':{},'released':False,'max_concurrent_study_gpus':1,'full_shards':32,
            'reader_question_evaluations':1288,'reader_question_budget_configurations':5152,'base_max_gpu_hours':97.5,'max_gpu_hours_with_two_requeues':292.5,'runtime_preflight':runtime}
        atomic(receipt_path,receipt)
    own_ids={v['job'] for v in receipt['jobs'].values()}
    live=subprocess.check_output(['squeue','-h','-u',getpass.getuser(),'-o','%i|%j|%T'],text=True).splitlines()
    foreign=[line for line in live if len(line.split('|'))==3 and line.split('|')[1].startswith('ctx-stab') and line.split('|')[0] not in own_ids]
    if foreign:raise RuntimeError('Another stability batch already exists; inspect before duplicating: '+repr(foreign))
    def submit(key,args,metadata):
        if key in receipt['jobs']:return receipt['jobs'][key]['job']
        job=subprocess.check_output(['sbatch','--parsable','--hold',*args],text=True).strip().split(';')[0]
        receipt['jobs'][key]={'job':job,**metadata};atomic(receipt_path,receipt)
        print(key,job,flush=True);return job
    common=['--account=tealab','--partition=a100_normal_q','--qos=tc_a100_normal_short','--gres=gpu:1','--cpus-per-task=1',
        '--mem=64G','--requeue','--signal=B:USR1@120','--open-mode=append']
    previous=None;smokes={};gpu_ids=[]
    for model in ('qwen7','mistral7'):
        dep=[f'--dependency=afterany:{previous}'] if previous else []
        smoke=submit('smoke/'+model,common+['--time=00:45:00',f'--job-name=ctx-stab-smoke-{model}','--output=logs/smoke_%j.log',*dep,'run_stability.slurm',model,'smoke'],{'kind':'smoke','model':model})
        smokes[model]=smoke;previous=smoke;gpu_ids.append(smoke)
    for model in ('qwen7','mistral7'):
        for dataset in ('musique','2wiki'):
            for shard in range(8):
                key=f'full/{model}/{dataset}/{shard}'
                job=submit(key,common+['--time=03:00:00',f'--job-name=ctx-stab-{model}','--output=logs/shard_%j.log',
                    f'--dependency=afterok:{smokes[model]},afterany:{previous}','--kill-on-invalid-dep=yes',
                    'run_stability.slurm',model,dataset,str(shard)],{'kind':'full','model':model,'dataset':dataset,'shard':shard})
                previous=job;gpu_ids.append(job)
    review=submit('review',['--account=tealab','--partition=normal_q','--cpus-per-task=2','--mem=8G','--time=00:30:00',
        '--job-name=ctx-stab-review','--output=logs/review_%j.log','--dependency=afterany:'+':'.join(gpu_ids),'run_stability_review.slurm'],{'kind':'review'})
    assert len(receipt['jobs'])==35 and len(gpu_ids)==34
    # Validate all newly staged IDs remain held before releasing the chain.
    all_ids=[v['job'] for v in receipt['jobs'].values()]
    states=subprocess.check_output(['squeue','-h','-j',','.join(all_ids),'-o','%i|%T|%r'],text=True).splitlines()
    assert len(states)==35 and all('|PENDING|JobHeldUser' in s for s in states),states
    subprocess.run(['scontrol','release',*all_ids],check=True)
    receipt['released']=True;atomic(receipt_path,receipt)
    print(json.dumps({'smokes':smokes,'review':review,'jobs':35,'released':True,'receipt':str(receipt_path)},indent=2),flush=True)


if __name__=='__main__':main()
