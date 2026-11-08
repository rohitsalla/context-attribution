"""Bounded real-HF gradient canary. Training loss is not generalization evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from .hf import HFModel
from .memory import Source
from .run import raw_memory
from .soft import SourceSoftMemory,paired_effect_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--encoder',required=True);p.add_argument('--encoder-revision',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=64)
    p.add_argument('--limit',type=int,default=8);p.add_argument('--slots',type=int,default=8)
    p.add_argument('--lambda-cf',type=float,default=1);args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise SystemExit('Canary output must be new')
    args.output.mkdir(parents=True,exist_ok=True)
    items=[json.loads(s) for s in args.dataset.read_text().splitlines()][:args.limit]
    if any(i['split']!='train' for i in items):raise ValueError('Training canary accepts training examples only')
    torch.manual_seed(1729)
    decoder=HFModel(args.model,args.revision);encoder=HFModel(args.encoder,args.encoder_revision)
    decoder.model.requires_grad_(False);encoder.model.requires_grad_(False)
    dec_embeddings=decoder.model.get_input_embeddings()
    rms=float(dec_embeddings.weight[:2048].float().square().mean().sqrt())
    enc_dim=encoder.model.config.hidden_size;dec_dim=decoder.model.config.hidden_size
    torch.manual_seed(1729)
    pool=SourceSoftMemory(enc_dim,dec_dim,args.slots,rms).to(decoder.device)
    initial_pool_sha=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in pool.state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW(pool.parameters(),lr=.001)
    features=[];truncated=0;tick=time.perf_counter()
    for item in items:
        values=[]
        for source in item['sources']:
            tokens=encoder.tokenizer.encode(source['title']+': '+source['text'],add_special_tokens=True)
            truncated+=len(tokens)>768;tokens=tokens[:768]
            ids=torch.tensor([tokens],device=encoder.device)
            with torch.no_grad():h=encoder.model.model(input_ids=ids,use_cache=False).last_hidden_state.detach()
            values.append(h)
        features.append(values)
    feature_seconds=time.perf_counter()-tick
    def soft_prefix(index,omit=None):
        item=items[index];left,right=decoder.qa_sides(item['question'])
        def embed(ids):return dec_embeddings(torch.tensor([ids],device=decoder.device))
        parts=[embed(left)]
        for k,(source,h) in enumerate(zip(item['sources'],features[index])):
            if k==omit:continue
            header=decoder.tokenizer.encode(f"\n[{source['id']}]\n",add_special_tokens=False)
            parts.extend([embed(header),pool(h,torch.ones(h.shape[:2],dtype=torch.bool,device=h.device)).to(dec_embeddings.weight.dtype)])
        return torch.cat(parts+[embed(right)],dim=1)
    def soft_logp(index,target,omit=None):
        prefix=soft_prefix(index,omit)
        ids=torch.tensor([target],device=decoder.device)
        combined=torch.cat([prefix,dec_embeddings(ids)],dim=1)
        logits=decoder.model(inputs_embeds=combined,use_cache=False).logits[0,prefix.shape[1]-1:-1].float()
        return logits.log_softmax(-1).gather(-1,ids[0,:,None]).sum()
    targets=[decoder.tokenizer.encode(i['gold'],add_special_tokens=False) for i in items]
    def factual_mean():
        with torch.no_grad():return sum(-float(soft_logp(j,t))/len(t) for j,t in enumerate(targets))/len(items)
    initial=factual_mean();teacher_cache={};trace=[]
    for step in range(args.steps):
        j=step%len(items);item=items[j];omit=(step//len(items))%len(item['sources']);target=targets[j]
        sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']]
        key=(j,omit)
        if key not in teacher_cache:
            factual=decoder.score(decoder.qa_ids(item['question'],raw_memory(decoder.tokenizer,sources)),target,stage='teacher_factual')
            changed=decoder.score(decoder.qa_ids(item['question'],raw_memory(decoder.tokenizer,[s for k,s in enumerate(sources) if k!=omit])),target,stage='teacher_intervention')
            teacher_cache[key]=(factual,changed)
        optimizer.zero_grad(set_to_none=True)
        student=torch.stack([soft_logp(j,target),soft_logp(j,target,omit)]).unsqueeze(0)
        teacher=torch.tensor([teacher_cache[key]],device=student.device)
        effect=paired_effect_loss(teacher/len(target),student/len(target))
        ce=-student[0,0]/len(target)
        factual_distill=(student[0,0]/len(target)-teacher[0,0]/len(target)).square()
        loss=ce+.01*factual_distill+args.lambda_cf*effect
        if not torch.isfinite(loss):raise ValueError('Nonfinite training canary loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(pool.parameters(),1.0)
        if not torch.isfinite(norm):raise ValueError('Nonfinite adapter gradients')
        optimizer.step()
        record={'step':step,'item_id':item['id'],'source_index':omit,'loss':float(loss.detach()),
                'factual_ce':float(ce.detach()),'paired_loss':float(effect.detach()),'gradient_norm':float(norm)}
        trace.append(record)
        if step%8==0:print(json.dumps(record),flush=True)
    final=factual_mean()
    torch.save(pool.state_dict(),args.output/'pool.pt')
    summary={'type':'training_canary_not_research_efficacy','steps':args.steps,'training_examples':len(items),
        'lambda_cf':args.lambda_cf,'slots_per_source':args.slots,'seed':1729,
        'initial_train_ce':initial,'final_train_ce':final,'training_ce_decreased':final<initial,
        'encoder_truncated_sources':truncated,'feature_encoding_seconds':feature_seconds,
        'trainable_parameters':sum(p.numel() for p in pool.parameters()),
        'initial_pool_sha256':initial_pool_sha,
        'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
        'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        'answerer_revision':args.revision,'encoder_revision':args.encoder_revision,
        'runtime':{'answerer':decoder.runtime(),'encoder':encoder.runtime()},
        'note':'Gold answers and paired raw-context teacher scores are training-only supervision. No dev/test efficacy is claimed.'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
