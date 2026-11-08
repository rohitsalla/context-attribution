"""Pinned HF inference with token-space memory insertion and exact target tokens."""
import time


class HFModel:
    def __init__(self,path,revision,*,model_id=None,device='cuda'):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        import os
        self.torch=torch;self.device=device;self.model_id=model_id or path;self.revision=revision
        torch.set_num_threads(max(1,min(8,int(os.environ.get('SLURM_CPUS_PER_TASK','1')))))
        self.tokenizer=AutoTokenizer.from_pretrained(path,revision=revision,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(path,revision=revision,
                        torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
        self.calls=[]

    def _sync(self):
        if self.device.startswith('cuda'):self.torch.cuda.synchronize()

    def chat_ids(self,text):
        rendered=self.tokenizer.apply_chat_template([{'role':'user','content':text}],
                         tokenize=False,add_generation_prompt=True,enable_thinking=False)
        return self.tokenizer.encode(rendered,add_special_tokens=False)

    def qa_sides(self,question):
        marker='<<<MEMORY_INSERTION_9f83>>>'
        text=('Use the supplied passages to answer the question. Return only the short answer, '
              'without explanation or citations. If evidence is insufficient, return UNKNOWN.\n'
              'Passages:\n'+marker+'\nQuestion: '+question+'\nAnswer:')
        rendered=self.tokenizer.apply_chat_template([{'role':'user','content':text}],
                         tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if rendered.count(marker)!=1:raise ValueError('ambiguous memory insertion marker')
        left,right=rendered.split(marker)
        return (self.tokenizer.encode(left,add_special_tokens=False),
                self.tokenizer.encode(right,add_special_tokens=False))

    def qa_ids(self,question,memory):
        left,right=self.qa_sides(question)
        return left+list(memory)+right

    def generate(self,prefix,*,cap=64,seed=1729,stage='generation'):
        torch=self.torch;torch.manual_seed(seed);self._sync();tick=time.perf_counter()
        ids=torch.tensor([prefix],device=self.device)
        with torch.inference_mode():
            result=self.model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=cap,
                         do_sample=False,pad_token_id=self.tokenizer.eos_token_id)[0,len(prefix):].tolist()
        self._sync();seconds=time.perf_counter()-tick
        self.calls.append({'stage':stage,'seconds':seconds,'input_tokens':len(prefix),'output_tokens':len(result)})
        special=set(self.tokenizer.all_special_ids)
        target=[i for i in result if i not in special]
        return {'tokens':target,'text':self.tokenizer.decode(target,skip_special_tokens=True),
                'cap_reached':len(result)>=cap}

    def score(self,prefix,target,*,stage='score',grad=False):
        if not target:raise ValueError('empty target span')
        torch=self.torch;self._sync();tick=time.perf_counter()
        ids=torch.tensor([list(prefix)+list(target)],device=self.device)
        with torch.set_grad_enabled(grad):
            logits=self.model(input_ids=ids,use_cache=False).logits[0,len(prefix)-1:-1].float()
            labels=ids[0,len(prefix):]
            value=logits.log_softmax(-1).gather(-1,labels[:,None]).sum()
        self._sync()
        self.calls.append({'stage':stage,'seconds':time.perf_counter()-tick,
                           'input_tokens':len(prefix),'output_tokens':len(target)})
        return value if grad else float(value.item())

    def cost(self,start=0):
        calls=self.calls[start:]
        return {'calls':len(calls),'seconds':sum(c['seconds'] for c in calls),
                'input_tokens':sum(c['input_tokens'] for c in calls),
                'output_tokens':sum(c['output_tokens'] for c in calls)}


class AbstractiveCodec:
    def __init__(self,compressor,answer_tokenizer):
        self.compressor=compressor;self.tokenizer=answer_tokenizer
        self.tag=f'qa-facts-v2:{compressor.model_id}@{compressor.revision}'

    def __call__(self,sources,quota,question,seed):
        text=('Compress these passages into terse factual notes useful for the question. '
              'Preserve names, relations, dates, and facts that may connect to another passage. '
              'Put a source ID such as [d0] after each fact. Output notes only, no introduction. '
              f'The output has a strict budget of approximately {quota} answer-model tokens.\n'
              f'Question: {question}\nPassages:\n'+
              '\n'.join(f'[{s.sid}] {s.title}: {s.text}' for s in sources))
        out=self.compressor.generate(self.compressor.chat_ids(text),cap=max(8,2*quota),seed=seed,stage='compression')
        ids=self.tokenizer.encode(out['text'],add_special_tokens=False)
        return ids[:quota],{'generated_answerer_tokens':len(ids),'clipped':len(ids)>quota,
                           'generation_cap_reached':out['cap_reached']}
