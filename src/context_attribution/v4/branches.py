"""Branch objectives from ONE shared warm-up, optimizer and random state."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from ..v2.hf import HFModel
from ..v2.memory import ExtractiveCodec,Source,digest
from ..v3.soft_study import Study,read_items,reference,state_hash,objective,evaluate,lexical_control


def tree_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=str):h.update(str(key).encode());visit(x[key])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def snapshot(pool,optimizer):
    return {'pool':copy.deepcopy(pool.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
            'cpu_rng':torch.get_rng_state().clone(),'cuda_rng':[x.clone() for x in torch.cuda.get_rng_state_all()]}


def restore(pool,optimizer,state):
    pool.load_state_dict(state['pool']);optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    actual={'pool':pool.state_dict(),'optimizer':optimizer.state_dict(),
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    expected={k:tree_hash(v) for k,v in state.items()};observed={k:tree_hash(v) for k,v in actual.items()}
    if expected!=observed:raise ValueError('Branch did not restore exact warm-up/optimizer/RNG state')
    return observed


class AnchoredStudy(Study):
    """Residual embedding edits with an exact lexical identity at gate=0.

    This is an established residual/gating construction, not a novelty claim.
    No extra memory positions versus the identical lexical control.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pool.register_parameter('residual_gate',torch.nn.Parameter(torch.zeros((),device=self.decoder.device)))
        self.literal={};self.rms=float(self.emb.weight[:2048].detach().float().square().mean().sqrt())

    def components(self,item):
        if item['id'] not in self.literal:
            codec=ExtractiveCodec(self.decoder.tokenizer,'lexical')
            self.literal[item['id']]=[codec([Source(s['id'],s['title'],s['text'])],self.slots,item['question'],1729)[0]
                                     for s in item['sources']]
        values=[]
        for h,ids in zip(self.encode(item)[0],self.literal[item['id']]):
            delta=self.pool(h,torch.ones(h.shape[:2],dtype=torch.bool,device=h.device))[:,:len(ids)]
            delta=delta/delta.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)*self.rms
            literal=self.emb(torch.tensor([ids],device=self.decoder.device))
            values.append(literal+(self.pool.residual_gate.tanh()*delta).to(literal.dtype))
        return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--architecture',choices=['pure','anchored'],required=True)
    p.add_argument('--seed',type=int,default=1729);p.add_argument('--train-limit',type=int,default=128)
    p.add_argument('--dev-limit',type=int,default=96);p.add_argument('--warmup-steps',type=int,default=256)
    p.add_argument('--steps',type=int,default=512);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Fresh suite output required')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.use_deterministic_algorithms(True)
    registry=json.loads((args.root/'research/v2/models.json').read_text());paths=json.loads(args.paths.read_text())
    decoder=HFModel(paths['qwen7'],registry['qwen7']['revision'],model_id=registry['qwen7']['model_id'])
    encoder=HFModel(paths['qwen15'],registry['qwen15']['revision'],model_id=registry['qwen15']['model_id'])
    decoder.model.generation_config.repetition_penalty=1.
    train=read_items(args.root/'data/v2-hotpot/train.jsonl','train',args.train_limit)
    dev=read_items(args.root/'data/v2-hotpot/dev.jsonl','dev',args.dev_limit)
    torch.manual_seed(args.seed)
    study=(AnchoredStudy if args.architecture=='anchored' else Study)(decoder,encoder,32)
    optimizer=torch.optim.AdamW(study.pool.parameters(),lr=.001)
    branches=[('factual',0.),('unpaired',.1),('paired',.1),('unpaired',1.),('paired',1.)]
    if args.smoke:branches=[('factual',0.),('paired',.1)]
    manifest={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest.update({'runtime':{'answerer':decoder.runtime(),'encoder':encoder.runtime()},
       'attention_backend':'forced SDPA math; deterministic algorithms; shared checkpoint, optimizer, RNG',
       'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
       'source_sha256':{str(f.relative_to(args.root)):hashlib.sha256(f.read_bytes()).hexdigest()
                       for folder in ['v2','v3','v4'] for f in (args.root/'src/context_attribution'/folder).glob('*.py')},
       'data_sha256':{name:hashlib.sha256((args.root/'data/v2-hotpot'/name).read_bytes()).hexdigest() for name in ['train.jsonl','dev.jsonl']},
       'branches':branches,'generation':{'greedy':True,'repetition_penalty':1.}})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with sdpa_kernel(SDPBackend.MATH):
        refs={item['id']:reference(item,decoder) for item in dev}
        (args.output/'raw_reference.json').write_text(json.dumps(refs))
        if args.architecture=='anchored':
            # Before any update, soft inputs must be the exact literal embedding
            # stream. Check in token space too, rather than assuming it.
            from ..v2.memory import Contract
            for item in dev[:2]:
                sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']]
                codec=ExtractiveCodec(decoder.tokenizer,'lexical')
                budget=sum(len(decoder.tokenizer.encode(f'\n[{s.sid}]\n',add_special_tokens=False))+32 for s in sources)
                policy=Contract(decoder.tokenizer,[s.sid for s in sources],mode='source',budget=budget,codec_tag=codec.tag,question=item['question'])
                memory=policy.build(sources,codec)
                with torch.no_grad():
                    prefix,_=study.prefix(item,study.components(item))
                    ids=decoder.qa_ids(item['question'],memory.tokens)
                    literal=study.emb(torch.tensor([ids],device=decoder.device))
                    if not torch.equal(prefix,literal):raise ValueError('Lexical anchor is not identity at initialization')
        targets={i['id']:decoder.tokenizer.encode(i['gold'],add_special_tokens=False) for i in train}
        teachers={}
        for item in train:
            study.encode(item);teachers[item['id']]=reference(item,decoder,targets[item['id']])
        initial_hash=state_hash(study.pool)
        def loss_at(step,mode,weight):
            epoch,index=divmod(step,len(train));order=list(range(len(train)));random.Random(args.seed+epoch).shuffle(order)
            item=train[order[index]];ref=teachers[item['id']];target=targets[item['id']];n=len(target)
            omit=(int(digest([args.seed,step,item['id']])[:8],16)%len(item['sources']),)
            components=study.components(item);factual,_=study.prefix(item,components);changed,_=study.prefix(item,components,omit)
            student=torch.stack([study.score(factual,target),study.score(changed,target)]).unsqueeze(0)/n
            teacher=torch.tensor([[ref['base'],ref['changed'][str(omit[0])]]],device=decoder.device)/n
            ce=-student[0,0];kd=(student[0,0]-teacher[0,0]).square();auxiliary=objective(student,teacher,mode)
            return ce+.01*kd+weight*auxiliary,{'step':step,'ce':float(ce.detach()),'auxiliary':float(auxiliary.detach()),'item_id':item['id']}
        def update(step,mode,weight):
            optimizer.zero_grad(set_to_none=True);loss,record=loss_at(step,mode,weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite branch loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(study.pool.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite branch gradients')
            optimizer.step();record.update({'loss':float(loss.detach()),'gradient_norm':float(norm)})
            return record
        # Detect the warm-up failure observed in V3 before spending on comparisons.
        gradients=[]
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True);loss,_=loss_at(0,'factual',0.);loss.backward()
            gradients.append(tree_hash({n:p.grad for n,p in study.pool.named_parameters()}))
        if gradients[0]!=gradients[1]:raise ValueError('Repeated same-state gradients differ under deterministic math attention')
        warmup=[update(step,'factual',0.) for step in range(args.warmup_steps)]
        shared=snapshot(study.pool,optimizer);shared_hashes={k:tree_hash(v) for k,v in shared.items()}
        torch.save(shared,args.output/'shared_warmup.pt')
        (args.output/'warmup.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in warmup))
        receipts={}
        lexical_control(decoder,dev,refs,args.output/'lexical32.jsonl',32)
        for mode,weight in branches:
            label=f'{mode}_{weight:g}';folder=args.output/label;folder.mkdir()
            receipts[label]=restore(study.pool,optimizer,shared)
            records=[update(args.warmup_steps+step,mode,weight) for step in range(args.steps)]
            (folder/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            torch.save(study.pool.state_dict(),folder/'pool.pt')
            evaluate(study,dev,refs,folder/'dev.jsonl',label)
            (folder/'completed.json').write_text(json.dumps({'complete':True,'initial_shared_state':receipts[label],
                                                           'final_pool_sha256':state_hash(study.pool)},indent=2))
        (args.output/'completed.json').write_text(json.dumps({'complete':True,'seed':args.seed,'architecture':args.architecture,
            'initial_pool_sha256':initial_hash,'shared_hashes':shared_hashes,'all_branch_start_states_equal':all(v==shared_hashes for v in receipts.values()),
            'same_state_gradient_hash':gradients[0],'branches':list(receipts),'dev_n':len(dev)},indent=2))


if __name__=='__main__':main()
