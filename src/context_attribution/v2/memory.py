"""Fixed-routing memory with explicit, checkable intervention dependencies.

This is an incremental-computation contract, not a novelty or QA guarantee.
The source universe, routing, per-component budgets and codec are fixed for
the factual and intervened branches. Source deletions never redistribute budget.
"""
from dataclasses import dataclass
import hashlib
import json
import re
import math
from collections import Counter


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Source:
    sid: str
    title: str
    text: str


@dataclass(frozen=True)
class Memory:
    tokens: tuple
    chunks: dict
    dependencies: dict
    source_hashes: dict
    contract: str
    diagnostics: dict


class Contract:
    def __init__(self,tokenizer,universe,*,mode,budget,groups=4,codec_tag='lead-v1',question='',seed=1729):
        self.tokenizer=tokenizer;self.universe=tuple(universe);self.mode=mode
        if len(set(universe))!=len(universe):raise ValueError('duplicate source IDs')
        if mode not in ['global','source','hash']:raise ValueError(mode)
        self.assignment={sid:('g0' if mode=='global' else sid if mode=='source' else
                             'g'+str(int(digest([sid,seed])[:12],16)%groups)) for sid in universe}
        self.order=tuple(dict.fromkeys(self.assignment.values()))
        self.headers={g:tuple(tokenizer.encode(f'\n[{g}]\n',add_special_tokens=False)) for g in self.order}
        available=budget-sum(map(len,self.headers.values()))
        if available<len(self.order):raise ValueError('budget cannot fit framing and one token per component')
        self.quotas={g:available//len(self.order)+(i<available%len(self.order)) for i,g in enumerate(self.order)}
        self.budget=budget;self.question=question;self.seed=seed
        self.fingerprint=digest({'universe':self.universe,'assignment':self.assignment,'headers':self.headers,
                                 'quotas':self.quotas,'codec':codec_tag,'question':question,'seed':seed,
                                 'tokenizer':getattr(tokenizer,'name_or_path',type(tokenizer).__qualname__)})

    def _inputs(self,sources):
        sources=list(sources)
        if len({s.sid for s in sources})!=len(sources):raise ValueError('duplicate current source IDs')
        if not {s.sid for s in sources}<=set(self.universe):raise ValueError('source outside fixed universe')
        by_id={s.sid:s for s in sources}
        ordered=[by_id[sid] for sid in self.universe if sid in by_id]
        return ordered,{s.sid:digest([s.title,s.text]) for s in ordered}

    def build(self,sources,codec,*,factual=None):
        ordered,hashes=self._inputs(sources)
        if factual is not None and factual.contract!=self.fingerprint:raise ValueError('memory contract mismatch')
        changed=set(hashes) if factual is None else {
            sid for sid in set(hashes)|set(factual.source_hashes) if hashes.get(sid)!=factual.source_hashes.get(sid)}
        affected={self.assignment[sid] for sid in changed}
        chunks={};deps={};diagnostics={};stream=[]
        for group in self.order:
            members=[s for s in ordered if self.assignment[s.sid]==group]
            deps[group]=tuple(s.sid for s in members)
            if not members:
                body=();info={'empty':True,'codec_called':False}
            elif factual is not None and group not in affected:
                body=factual.chunks[group];info={'reused':True,'codec_called':False}
            else:
                component_seed=int(digest([self.seed,group,[(s.sid,hashes[s.sid]) for s in members]])[:8],16)
                body,info=codec(members,self.quotas[group],self.question,component_seed)
                if len(body)>self.quotas[group]:raise ValueError('codec exceeded component quota')
                body=tuple(body);info={**info,'codec_called':True}
            chunks[group]=body;diagnostics[group]=info
            if members:stream.extend(self.headers[group]);stream.extend(body)
        if len(stream)>self.budget:raise AssertionError('compiled memory exceeds budget')
        return Memory(tuple(stream),chunks,deps,hashes,self.fingerprint,diagnostics)


class ExtractiveCodec:
    def __init__(self,tokenizer,method='lead'):
        if method not in ['lead','lexical']:raise ValueError(method)
        self.tokenizer=tokenizer;self.method=method;self.tag=method+'-source-local-v1'

    def __call__(self,sources,quota,question,seed):
        text='\n'.join(s.title+': '+s.text for s in sources)
        if self.method=='lexical':
            # Sentence ranking is fitted only inside this component, so changing
            # another component cannot change its IDF or its cached representation.
            from ..attribution import sentences
            parts=sentences(text) or [text]
            bags=[Counter(re.findall(r'\w+',p.lower())) for p in parts]
            query=set(re.findall(r'\w+',question.lower()));n=len(parts)
            average=sum(sum(b.values()) for b in bags)/max(n,1)
            def score(j):
                bag=bags[j];length=sum(bag.values());total=0
                for word in query:
                    df=sum(word in x for x in bags);idf=math.log(1+(n-df+.5)/(df+.5))
                    tf=bag[word];den=tf+1.2*(.25+.75*length/max(average,1))
                    total+=idf*tf*2.2/den if den else 0
                return total
            chosen=sorted(range(n),key=lambda j:(-score(j),j))
            text=' '.join(parts[j] for j in chosen)
        tokens=self.tokenizer.encode(text,add_special_tokens=False)
        return tokens[:quota],{'input_tokens':len(tokens),'clipped':len(tokens)>quota,'method':self.method}
