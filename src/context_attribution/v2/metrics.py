"""Answer and intervention metrics kept separate from semantic provenance."""
from collections import Counter
import re
import string
import numpy as np


def normalize(text):
    text=''.join(c for c in text.lower() if c not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def qa(prediction,gold):
    a,b=normalize(prediction),normalize(gold)
    exact=float(a==b)
    if a!=b and ({a,b}&{'yes','no','noanswer'}):return {'em':exact,'f1':0.0}
    common=sum((Counter(a.split())&Counter(b.split())).values())
    f1=2*common/(len(a.split())+len(b.split())) if a and b else exact
    return {'em':exact,'f1':f1}


def effect_comparison(reference,estimated):
    ids=sorted(reference);a=np.array([reference[s] for s in ids],float)
    b=np.array([estimated[s] for s in ids],float)
    scale=float(np.abs(a).sum());selected=int(np.argmax(np.abs(b)))
    return {'mean_absolute_error':float(np.mean(np.abs(a-b))),
            'normalized_absolute_error':float(np.abs(a-b).sum()/max(scale,1e-8)),
            'reference_l1':scale,'reference_negligible':bool(np.max(np.abs(a))<=1e-5),
            'top1_hit':None if np.max(np.abs(a))<=1e-5 else bool(np.abs(a).max()-abs(a[selected])<=1e-5)}
