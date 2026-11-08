import unittest
from context_attribution.v2.memory import Contract,Source,ExtractiveCodec


class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False):return list(text.encode())
    def decode(self,tokens):return bytes(tokens).decode()


class RecordingCodec:
    def __init__(self):self.calls=[]
    def __call__(self,sources,quota,question,seed):
        self.calls.append(tuple(s.sid for s in sources))
        return list(('|'.join(s.title+':'+s.text for s in sources)+question).encode())[:quota],{}


class MemoryContractTests(unittest.TestCase):
    def setUp(self):
        self.tok=CharacterTokenizer()
        self.docs=[Source(f'd{i}',f'Title {i}',f'Fact {i} with bridge evidence.') for i in range(10)]

    def test_all_deletions_match_uncached_reference_and_budget(self):
        for mode in ['global','source','hash']:
            policy=Contract(self.tok,[d.sid for d in self.docs],mode=mode,budget=192)
            codec=RecordingCodec();factual=policy.build(self.docs,codec)
            self.assertLessEqual(len(factual.tokens),192)
            for removed in self.docs:
                kept=[d for d in self.docs if d.sid!=removed.sid]
                codec.calls.clear();fast=policy.build(kept,codec,factual=factual)
                calls=len(codec.calls);cold=policy.build(kept,RecordingCodec())
                self.assertEqual(fast.tokens,cold.tokens)
                self.assertLessEqual(len(fast.tokens),192)
                self.assertLessEqual(calls,1)
                if mode=='source':self.assertEqual(calls,0)
                if mode=='global':self.assertEqual(calls,1)

    def test_replacement_replays_only_affected_component(self):
        policy=Contract(self.tok,[d.sid for d in self.docs],mode='hash',budget=192)
        factual=policy.build(self.docs,RecordingCodec());changed=list(self.docs)
        changed[2]=Source('d2','Replacement','A different fact')
        codec=RecordingCodec();updated=policy.build(changed,codec,factual=factual)
        self.assertEqual(len(codec.calls),1)
        self.assertEqual(updated.tokens,policy.build(changed,RecordingCodec()).tokens)

    def test_different_contract_is_rejected(self):
        p=Contract(self.tok,[d.sid for d in self.docs],mode='source',budget=192,question='A')
        q=Contract(self.tok,[d.sid for d in self.docs],mode='source',budget=192,question='B')
        with self.assertRaises(ValueError):q.build(self.docs,RecordingCodec(),factual=p.build(self.docs,RecordingCodec()))

    def test_invalid_budget_or_universe_is_not_silently_changed(self):
        with self.assertRaises(ValueError):Contract(self.tok,['d0','d0'],mode='source',budget=192)
        with self.assertRaises(ValueError):Contract(self.tok,[d.sid for d in self.docs],mode='source',budget=8)
        policy=Contract(self.tok,['d0'],mode='source',budget=32)
        with self.assertRaises(ValueError):policy.build(self.docs,RecordingCodec())

    def test_lexical_codec_remains_source_local_under_deletion(self):
        policy=Contract(self.tok,[d.sid for d in self.docs],mode='source',budget=192,question='bridge')
        codec=ExtractiveCodec(self.tok,'lexical');factual=policy.build(self.docs,codec)
        fast=policy.build(self.docs[1:],codec,factual=factual)
        cold=policy.build(self.docs[1:],codec)
        self.assertEqual(fast.tokens,cold.tokens)
