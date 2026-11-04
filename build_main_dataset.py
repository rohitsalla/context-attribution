"""Build the frozen main dataset and explicitly labeled stress-test subsets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import random
import re
import urllib.request

API = 'https://api.nobelprize.org/2.1/laureates?limit=200&sort=asc&nobelPrizeYear=2000&yearTo=2015'
ROOT = Path(__file__).resolve().parent


class Text(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]; self.skip=0
    def handle_starttag(self, tag, attrs):
        if tag in ['script','style']: self.skip+=1
    def handle_endtag(self, tag):
        if tag in ['script','style']: self.skip-=1
    def handle_data(self, value):
        if not self.skip: self.parts.append(value)


def normalize(text):
    return ' '.join(text.split()).replace('\u2013','-').replace('\u2011','-')


def fetch(url, path):
    if not path.exists():
        req=urllib.request.Request(url, headers={'User-Agent':'ContextAttributionResearch/0.1'})
        with urllib.request.urlopen(req, timeout=45) as response: path.write_bytes(response.read())
    return path.read_bytes()


def natural_records(cache):
    raw=fetch(API,cache/'nobel-2000-2015.json')
    records=[]
    for laureate in json.loads(raw)['laureates']:
        for prize in laureate.get('nobelPrizes',[]):
            affiliations=prize.get('affiliations',[])
            if len(affiliations)!=1 or not 2000<=int(prize['awardYear'])<=2015: continue
            a=affiliations[0]; org=a['name']['en']; city=a.get('city',{}).get('en','')
            if not city or city.split(',')[0].casefold() in org.casefold(): continue
            if org in ['University of California','University of Illinois']: continue
            urls=[l['href'] for l in prize.get('links',[]) if l.get('class')==['laureate facts']]
            if not urls: continue
            records.append({'id':laureate['id'],'name':laureate.get('knownName',laureate.get('fullName'))['en'],
                            'year':prize['awardYear'],'category':prize['category']['en'], 'org':org,
                            'city':city,'country':a.get('country',{}).get('en',''),'url':urls[0]})
    records=sorted(records,key=lambda x:(x['year'],x['name']))[:40]
    assert len(records)==40
    def audit(record):
        body=fetch(record['url'],cache/f"nobel-{record['id']}.html")
        parser=Text(); parser.feed(body.decode('utf-8'))
        text=normalize(' '.join(parser.parts)); org=normalize(record['org']); city=normalize(record['city'])
        # Keep exact factual spans and their offsets in normalized official-page text.
        label='Affiliation at the time of the award:'
        begin=text.find(label)
        if begin<0: raise ValueError(f"Missing affiliation label: {record['name']}")
        inst=text.find(org,begin); place=text.find(city,inst)
        if not begin<=inst<place<begin+len(label)+len(org)+len(city)+80:
            raise ValueError(f"API/page affiliation mismatch: {record['name']}")
        if normalize(record['name']) not in text:
            raise ValueError(f"API/page name mismatch: {record['name']}")
        return {**record,'affiliation_span':text[begin:inst+len(org)],
                'location_span':text[inst:place+len(city)],
                'page_sha256':hashlib.sha256(body).hexdigest(),
                'normalized_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'affiliation_offsets':[begin,inst+len(org)],'location_offsets':[inst,place+len(city)],
                'audit_method':'official API record cross-checked against official facts-page text; assistant-reviewed table',
                'human_audited':False}
    with ThreadPoolExecutor(max_workers=4) as pool: return list(pool.map(audit,records))


TEMPLATES = [
 ('Project {p} is led by {n}.','{n} works in {c}.','In which city does the lead of Project {p} work?'),
 ('The coordinator of Project {p} is {n}.','The workplace of {n} is in {c}.','Project {p} has a coordinator. What is their workplace city?'),
 ('Project {p} appointed {n} as its director.','{n} has an office in {c}.','Where is the office of the director appointed by Project {p}?'),
 ('The researcher responsible for Project {p} is {n}.','{n} is employed in {c}.','Give the city where the researcher responsible for Project {p} is employed.'),
 ('Project {p} is managed by {n}.','{n} is based in {c}.','Which city is the base of the person managing Project {p}?'),
 ('The principal investigator for Project {p} is {n}.','The research office of {n} is located in {c}.','Name the city containing the research office of Project {p}\'s principal investigator.'),
 ('Project {p} names {n} as its curator.','{n} carries out their work in {c}.','Project {p} names a curator. In what city does that curator work?'),
 ('The supervisor assigned to Project {p} is {n}.','The home office of {n} is in {c}.','What is the home-office city of the supervisor assigned to Project {p}?'),
]


def fictional_items():
    colors=['Amber','Silver','Indigo','Copper','Ivory','Violet','Scarlet','Golden']
    surnames=['Alder','Birch','Cedar','Dahlia','Elm','Fern','Grove','Hazel','Iris','Juniper']
    first=['Mira','Niko','Tessa','Arun','Lena','Omar','Sora','Ilan']
    endings=['haven','ford','brook','mere','vale','port','wick','dale','crest','field']
    result=[]
    for f,template in enumerate(TEMPLATES):
        for j in range(10):
            def names(k):
                return {'p':colors[f]+' '+surnames[k], 'n':first[f]+' '+surnames[k],
                        'c':colors[f]+endings[k]}
            a,b=names(j),names((j+1)%10)
            blocks=[{'id':'s1','text':template[0].format(**a)}, {'id':'s2','text':template[1].format(**a)},
                    {'id':'s3','text':template[0].format(**b)}, {'id':'s4','text':template[1].format(**b)},
                    {'id':'s5','text':f"Project {a['p']} keeps its publicity materials in a blue cabinet."},
                    {'id':'s6','text':'The visitor cafe closes at sunset and serves tea.'}]
            random.Random(2901+f*10+j).shuffle(blocks)
            result.append({'id':f'syn-{f:02}-{j:02}','base_id':f'syn-{f:02}-{j:02}',
                'family':f'fictional-template-{f}', 'split':'dev' if f<4 else 'test',
                'question':template[2].format(**a),'gold':a['c'],'gold_aliases':[],
                'sources':blocks,'support_ids':['s1','s2'],'irrelevant_id':'s6',
                'data_kind':'controlled_fictional','audited_natural_text':False,
                'variant':'base','entities':a,'distractor_entities':b})
    return result


def real_items(records):
    result=[]
    for i,a in enumerate(records):
        b=next(records[(i+k)%len(records)] for k in range(1,len(records))
               if records[(i+k)%len(records)]['org']!=a['org'] and records[(i+k)%len(records)]['city']!=a['city'])
        def source(sid, text, record):
            return {'id':sid,'text':text,'url':record['url'],'source_sha256':record['page_sha256']}
        blocks=[source('s1',a['name']+'\n[...]\n'+a['affiliation_span'],a),
                source('s2',a['location_span'],a),
                source('s3',b['name']+'\n[...]\n'+b['affiliation_span'],b),
                source('s4',b['location_span'],b),
                source('s5',f"{a['name']}\n[...]\n{a['category']} {a['year']}",a),
                source('s6',f"{b['name']}\n[...]\n{b['category']} {b['year']}",b)]
        random.Random(3901+i).shuffle(blocks)
        city=a['city'].split(',')[0]
        result.append({'id':f'natural-{i:02}','base_id':f'natural-{i:02}',
            'family':'nobel-affiliation-facts','split':'test',
            'question':f"In which city was {a['name']}'s institution located at the time of the {a['year']} prize? Give the city, omitting the state or country.",
            'gold':city,'gold_aliases':[a['city']],'sources':blocks,'support_ids':['s1','s2'],
            'irrelevant_id':'s6','data_kind':'natural_text_facts_page','audited_natural_text':True,
            'audit_method':a['audit_method'],'human_audited':False,'variant':'base'})
    return result


def stress_items(base):
    result=[]
    # Twelve prespecified fictional bases, balanced across development and test.
    selected=[r for r in base if r['data_kind']=='controlled_fictional'][::7][:12]
    for original in selected:
        for variant in ['paraphrase','lexical_distractor','duplicate','fact_replacement','misleading']:
            r=copy.deepcopy(original); r['id']+='-'+variant; r['variant']=variant
            a=r['entities']; by_id={s['id']:s for s in r['sources']}
            if variant=='paraphrase':
                by_id['s1']['text']=f"Responsibility for leading Project {a['p']} belongs to {a['n']}."
                by_id['s2']['text']=f"The city where {a['n']} does their work is {a['c']}."
            elif variant=='lexical_distractor':
                by_id['s5']['text']=f"An obsolete draft names {r['distractor_entities']['c']} as the workplace of Project {a['p']}'s lead. The draft was withdrawn and is not the current record."
            elif variant=='duplicate':
                by_id['s6']['text']=by_id['s2']['text'];r['support_ids']=['s1','s2','s6'];r['irrelevant_id']='s5'
                r['diagnostic_groups']=[['s2','s6']]
            elif variant=='fact_replacement':
                replacement='Azureharbor'
                for s in r['sources']: s['text']=s['text'].replace(a['c'],replacement)
                r['question']=r['question'].replace(a['c'],replacement)
                r['gold']=replacement;r['counterfactual_of']=original['id']
            elif variant=='misleading':
                by_id['s2']['text']=by_id['s2']['text'].replace(a['c'],r['distractor_entities']['c'])
                r['support_ids']=['s1'];r['truth_gold']=original['gold'];r['corrupted_source_ids']=['s2']
            result.append(r)
    return result


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data/main-v1')
    args=p.parse_args();cache=ROOT/'data/source-cache';cache.mkdir(parents=True,exist_ok=True)
    if args.output.exists(): raise SystemExit('Refusing to overwrite a frozen dataset directory')
    records=natural_records(cache)
    args.output.mkdir(parents=True)
    base=fictional_items()+real_items(records); stress=stress_items(base)
    from context_attribution.data import validate
    validate(base);validate(stress)
    assert len(base)==120 and sum(r['split']=='dev' for r in base)==40
    write_jsonl(args.output/'main.jsonl',base);write_jsonl(args.output/'stress.jsonl',stress)
    stochastic=[base[i] for i in [0,10,20,30,40,50,60,70]]
    write_jsonl(args.output/'stochastic.jsonl',stochastic)
    (args.output/'natural_audit.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()}
    (args.output/'manifest.json').write_text(json.dumps({'version':'main-v1','base_items':120,
        'fictional_items':80,'natural_facts_page_items':40,'dev_items':40,'test_items':80,
        'stress_items':len(stress),'stochastic_items':len(stochastic),'files':files,
        'natural_api':API,'human_audited':False,'natural_audit_status':'assistant-reviewed; human audit not claimed',
        'notes':['Natural subset consists of official facts-page excerpts, not long-form prose.',
                 'All natural examples are test-only; fictional templates 0-3 are development-only.']},indent=2))
    for r in records: print(r['name'],'|',r['org'],'|',r['city'],flush=True)
    print('Frozen',args.output,flush=True)


if __name__=='__main__': main()
