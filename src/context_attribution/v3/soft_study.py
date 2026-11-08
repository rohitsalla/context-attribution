"""Held-out development evaluation and matched learning-objective comparisons.

No locked-test access. Gold supervision is training-only; compression gets no
support labels or gold answers. All audit effects use the same raw-answer span.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from ..v2.hf import HFModel
from ..v2.memory import Contract,ExtractiveCodec,Source,digest
from ..v2.metrics import qa,effect_comparison
from ..v2.run import raw_memory
from ..v2.soft import SourceSoftMemory,paired_effect_loss


def read_items(path,split,limit):
    items=[json.loads(s) for s in path.read_text().splitlines()][:limit]
    if not items or any(i['split']!=split for i in items):raise ValueError('Unexpected/empty split')
    return items


def interventions(item,pairs=3):
    singleton=[(i,) for i in range(len(item['sources']))]
    joint=sorted(itertools.combinations(range(len(item['sources'])),2),
                 key=lambda pair:digest([item['id'],pair,'joint-dev-v1']))[:pairs]
    return singleton+joint


def objective(student,teacher,mode):
    """Same factual and perturbation forwards in all three controls."""
    if mode=='factual':return student.sum()*0
    if mode=='paired':return paired_effect_loss(teacher,student)
    if mode=='unpaired':return (student[:,1:]-teacher[:,1:]).square().mean()
    raise ValueError(mode)


class Study:
    def __init__(self,decoder,encoder,slots):
        self.decoder=decoder;self.encoder=encoder;self.slots=slots
        self.emb=decoder.model.get_input_embeddings();self.cache={};self.truncations=0
        self.pool=SourceSoftMemory(encoder.model.config.hidden_size,decoder.model.config.hidden_size,slots,
             float(self.emb.weight[:2048].float().square().mean().sqrt())).to(decoder.device)
        decoder.model.requires_grad_(False);encoder.model.requires_grad_(False)

    def sync(self):self.decoder._sync()

    def encode(self,item):
        if item['id'] in self.cache:return self.cache[item['id']]
        self.sync();tick=time.perf_counter();values=[]
        for source in item['sources']:
            ids=self.encoder.tokenizer.encode(source['title']+': '+source['text'],add_special_tokens=True)
            self.truncations+=len(ids)>768
            with torch.no_grad():
                h=self.encoder.model.model(input_ids=torch.tensor([ids[:768]],device=self.encoder.device),use_cache=False).last_hidden_state.detach()
            values.append(h)
        self.sync();value=(values,time.perf_counter()-tick);self.cache[item['id']]=value
        return value

    def components(self,item):
        return [self.pool(h,torch.ones(h.shape[:2],dtype=torch.bool,device=h.device)).to(self.emb.weight.dtype)
                for h in self.encode(item)[0]]

    def prefix(self,item,components,omit=()):
        def embed(ids):return self.emb(torch.tensor([ids],device=self.decoder.device))
        left,right=self.decoder.qa_sides(item['question']);parts=[embed(left)];positions=0
        for k,(source,value) in enumerate(zip(item['sources'],components)):
            if k in omit:continue
            header=self.decoder.tokenizer.encode(f"\n[{source['id']}]\n",add_special_tokens=False)
            parts.extend([embed(header),value]);positions+=len(header)+value.shape[1]
        return torch.cat(parts+[embed(right)],dim=1),positions

    def score(self,prefix,target):
        if not target:raise ValueError('Empty evaluation target')
        ids=torch.tensor([target],device=self.decoder.device)
        combined=torch.cat([prefix,self.emb(ids)],dim=1)
        logits=self.decoder.model(inputs_embeds=combined,use_cache=False).logits[0,prefix.shape[1]-1:-1].float()
        return logits.log_softmax(-1).gather(-1,ids[0,:,None]).sum()

    def generate(self,prefix):
        ids=self.decoder.model.generate(inputs_embeds=prefix,attention_mask=torch.ones(prefix.shape[:2],device=prefix.device,dtype=torch.long),
                max_new_tokens=64,do_sample=False,pad_token_id=self.decoder.tokenizer.eos_token_id)[0].tolist()
        tokens=[i for i in ids if i not in self.decoder.tokenizer.all_special_ids]
        return {'tokens':tokens,'text':self.decoder.tokenizer.decode(tokens,skip_special_tokens=True),'cap_reached':len(ids)>=64}


def reference(item,decoder,target=None):
    sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']]
    if target is None:answer=decoder.generate(decoder.qa_ids(item['question'],raw_memory(decoder.tokenizer,sources)),stage='dev_raw_answer')
    else:answer={'tokens':target,'text':decoder.tokenizer.decode(target),'training_gold_target':True}
    target=answer['tokens'];base=decoder.score(decoder.qa_ids(item['question'],raw_memory(decoder.tokenizer,sources)),target)
    values={}
    for omit in interventions(item):
        kept=[s for k,s in enumerate(sources) if k not in omit]
        values[','.join(map(str,omit))]=decoder.score(decoder.qa_ids(item['question'],raw_memory(decoder.tokenizer,kept)),target)
    return {'item_id':item['id'],'answer':answer,'base':base,'changed':values,'qa':qa(answer['text'],item['gold'])}


def state_hash(pool):
    return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in pool.state_dict().values())).hexdigest()


def lexical_control(decoder,items,references,output,slots):
    """Same source frames and per-source content-position allowance as soft memory."""
    codec=ExtractiveCodec(decoder.tokenizer,'lexical')
    with output.open('w') as handle:
        for item in items:
            sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']]
            budget=sum(len(decoder.tokenizer.encode(f'\n[{s.sid}]\n',add_special_tokens=False))+slots for s in sources)
            policy=Contract(decoder.tokenizer,[s.sid for s in sources],mode='source',budget=budget,
                            codec_tag=codec.tag,question=item['question'])
            memory=policy.build(sources,codec);ref=references[item['id']];target=ref['answer']['tokens'];n=len(target)
            answer=decoder.generate(decoder.qa_ids(item['question'],memory.tokens))
            base=decoder.score(decoder.qa_ids(item['question'],memory.tokens),target);effects={}
            for omit in interventions(item):
                changed=policy.build([s for k,s in enumerate(sources) if k not in omit],codec,factual=memory)
                effects[','.join(map(str,omit))]=(base-decoder.score(decoder.qa_ids(item['question'],changed.tokens),target))/n
            truth={k:(ref['base']-v)/n for k,v in ref['changed'].items()}
            single={k:v for k,v in effects.items() if ',' not in k};joint={k:v for k,v in effects.items() if ',' in k}
            row={'item_id':item['id'],'method':f'lexical_slots{slots}','answer':answer,'qa':qa(answer['text'],item['gold']),
                 'raw_qa':ref['qa'],'raw_answer':ref['answer'],'memory_positions':len(memory.tokens),'budget':budget,
                 'raw_effects':truth,'effects':effects,
                 'single_comparison':effect_comparison({k:truth[k] for k in single},single),
                 'joint_comparison':effect_comparison({k:truth[k] for k in joint},joint)}
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()


def evaluate(study,items,references,output,label):
    if output.exists():raise ValueError('Evaluation output already exists')
    study.pool.eval()
    with output.open('w') as handle,torch.no_grad():
        for item in items:
            ref=references[item['id']];target=ref['answer']['tokens'];n=len(target)
            features,encoding_seconds=study.encode(item);study.sync();tick=time.perf_counter()
            components=study.components(item);prefix,positions=study.prefix(item,components)
            answer=study.generate(prefix);base=float(study.score(prefix,target));effects={};single={};joint={}
            receipts=[]
            for index,omit in enumerate(interventions(item)):
                key=','.join(map(str,omit));changed,_=study.prefix(item,components,omit)
                value=float(study.score(changed,target));effects[key]=(base-value)/n
                (single if len(omit)==1 else joint)[key]=effects[key]
                if index<2:
                    cold,_=study.prefix(item,study.components(item),omit)
                    error=abs(float(study.score(cold,target))-value)
                    equal=torch.equal(changed,cold)
                    if not equal or error>1e-4:raise ValueError('Soft-memory cold replay mismatch')
                    receipts.append({'omitted':omit,'embedding_equal':equal,'score_error':error})
            study.sync();audit_seconds=time.perf_counter()-tick
            truth={k:(ref['base']-v)/n for k,v in ref['changed'].items()}
            row={'item_id':item['id'],'method':label,'qa':qa(answer['text'],item['gold']),
                 'answer':answer,'raw_qa':ref['qa'],'raw_answer':ref['answer'],'memory_positions':positions,
                 'common_target_tokens':target,'raw_effects':truth,'effects':effects,
                 'single_comparison':effect_comparison({k:truth[k] for k in single},single),
                 'joint_comparison':effect_comparison({k:truth[k] for k in joint},joint),
                 'cold_receipts':receipts,'source_encoding_seconds':encoding_seconds,
                 'evaluation_with_cold_diagnostics_seconds':audit_seconds,
                 'scope':'development only; timing includes cold diagnostics, not production cost'}
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
            print(label,item['id'],'F1',row['qa']['f1'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['canaries','factual','unpaired','paired'],required=True)
    p.add_argument('--seed',type=int,default=1729);p.add_argument('--slots',type=int,default=32)
    p.add_argument('--train-limit',type=int,default=128);p.add_argument('--dev-limit',type=int,default=32)
    p.add_argument('--warmup-steps',type=int,default=256);p.add_argument('--steps',type=int,default=512)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Require fresh study output')
    args.output.mkdir(parents=True,exist_ok=True)
    registry=json.loads((args.root/'research/v2/models.json').read_text());paths=json.loads(args.paths.read_text())
    decoder=HFModel(paths['qwen7'],registry['qwen7']['revision'],model_id=registry['qwen7']['model_id'])
    encoder=HFModel(paths['qwen15'],registry['qwen15']['revision'],model_id=registry['qwen15']['model_id'])
    # With inputs_embeds, token-based repetition penalties cannot inspect the
    # soft prompt. Disable them for ALL raw, lexical and soft generations.
    decoder.model.generation_config.repetition_penalty=1.0
    dev=read_items(args.root/'data/v2-hotpot/dev.jsonl','dev',args.dev_limit)
    torch.manual_seed(args.seed);study=Study(decoder,encoder,8 if args.mode=='canaries' else args.slots)
    manifest={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest.update({'runtime':{'answerer':decoder.runtime(),'encoder':encoder.runtime()},
       'generation_policy':{'do_sample':False,'repetition_penalty':1.0,'max_new_tokens':64},
       'source_sha256':{str(f.relative_to(args.root)):hashlib.sha256(f.read_bytes()).hexdigest()
                        for folder in ['v2','v3'] for f in (args.root/'src/context_attribution'/folder).glob('*.py')},
       'data_sha256':{name:hashlib.sha256((args.root/'data/v2-hotpot'/name).read_bytes()).hexdigest() for name in ['train.jsonl','dev.jsonl']},
       'split':'development only; locked test not loaded'})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    refs={item['id']:reference(item,decoder) for item in dev}
    (args.output/'raw_reference.json').write_text(json.dumps(refs))
    if args.mode=='canaries':
        for label in ['train_cf0','train_cf1']:
            path=args.root/'results/v2-dev'/label/'pool.pt'
            study.pool.load_state_dict(torch.load(path,map_location=decoder.device,weights_only=True))
            evaluate(study,dev,refs,args.output/f'{label}.jsonl',label)
        lexical_control(decoder,dev,refs,args.output/'lexical_slots8.jsonl',8)
        lexical_control(decoder,dev,refs,args.output/'lexical_slots32.jsonl',32)
        (args.output/'completed.json').write_text(json.dumps({'complete':True,'dev_n':len(dev)}));return
    train=read_items(args.root/'data/v2-hotpot/train.jsonl','train',args.train_limit)
    targets={item['id']:decoder.tokenizer.encode(item['gold'],add_special_tokens=False) for item in train}
    teacher={}
    # Teacher computations and perturbation forwards are identical across modes.
    for item in train:
        study.encode(item);teacher[item['id']]=reference(item,decoder,targets[item['id']])
    initial_hash=state_hash(study.pool);optimizer=torch.optim.AdamW(study.pool.parameters(),lr=.001)
    trace=[];warmup_hash=None
    for step in range(args.warmup_steps+args.steps):
        epoch,index=divmod(step,len(train));order=list(range(len(train)));random.Random(args.seed+epoch).shuffle(order)
        item=train[order[index]];ref=teacher[item['id']];target=targets[item['id']];n=len(target)
        omit=(int(digest([args.seed,step,item['id']])[:8],16)%len(item['sources']),)
        optimizer.zero_grad(set_to_none=True);components=study.components(item)
        factual,_=study.prefix(item,components);changed,_=study.prefix(item,components,omit)
        student=torch.stack([study.score(factual,target),study.score(changed,target)]).unsqueeze(0)/n
        teaching=torch.tensor([[ref['base'],ref['changed'][str(omit[0])]]],device=decoder.device)/n
        mode='factual' if step<args.warmup_steps else args.mode
        ce=-student[0,0];kd=(student[0,0]-teaching[0,0]).square()
        auxiliary=objective(student,teaching,mode);loss=ce+.01*kd+auxiliary
        if not torch.isfinite(loss):raise ValueError('Nonfinite study loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(study.pool.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite study gradient')
        optimizer.step()
        record={'step':step,'phase':mode,'item_id':item['id'],'omit':omit,'ce':float(ce.detach()),'auxiliary':float(auxiliary.detach()),'loss':float(loss.detach()),'gradient_norm':float(norm)}
        trace.append(record)
        if step+1==args.warmup_steps:warmup_hash=state_hash(study.pool)
        if step%64==0:print(json.dumps(record),flush=True)
    torch.save(study.pool.state_dict(),args.output/'pool.pt')
    (args.output/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
    evaluate(study,dev,refs,args.output/'dev.jsonl',args.mode)
    (args.output/'completed.json').write_text(json.dumps({'complete':True,'mode':args.mode,'seed':args.seed,
        'train_n':len(train),'dev_n':len(dev),'initial_pool_sha256':initial_hash,'warmup_pool_sha256':warmup_hash,
        'final_pool_sha256':state_hash(study.pool),'truncated_source_count':study.truncations},indent=2))


if __name__=='__main__':main()
