"""Two export products: frozen questions, or questions plus reviewed teacher training rows.

No model requests occur here. Collection adapters consume the frozen question set and
return independently reviewed, canonically packed runs. Models never share a train file.
"""
from pathlib import Path
import collections,hashlib,json,re,shutil
from urllib.parse import urlsplit
from pipeline.datakit import require_data_root

def digest(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_bytes(encoded(value))
def question_identity(bundle):
    t=bundle['task']
    return digest(encoded({'task':t,'followups':bundle.get('followups',[])}))
def profile_identity(profile):
    # Keys and credential locations do not enter dataset metadata or identity.
    fields=('provider','model','baseUrl','api','thinking','contextWindow','maxTokens','samplingParams','compat','systemPromptProfile','appendSystemPrompt','promptSuffix','protocol','adapterRevision','runtimeRevision','studentProfile')
    selected={k:profile[k]for k in fields if k in profile}
    for field in ('provider','model','baseUrl','api','thinking','protocol'):assert selected.get(field),field
    url=urlsplit(selected['baseUrl']);assert url.scheme in ('https','http') and url.hostname and not (url.username or url.password or url.query or url.fragment),'Endpoint must not contain credentials'
    assert selected['protocol']=='canonical XML+JSON','Explicit protocol migration required'
    return digest(encoded(selected)),selected
def fresh(out):
    out=require_data_root(Path(out));out.mkdir(parents=True,exist_ok=False);return out

def export_questions(sources,out):
    selected={}
    for source in sources:
        for p in sorted(Path(source).glob('*.json')):
            raw=p.read_bytes();b=json.loads(raw);t=b['task'];qid=t['id']
            assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}',qid),'Unsafe question ID'
            assert t['schemaVersion']==1 and t.get('source') and t.get('tools')
            assert t['split']=='train' and t.get('files') and t.get('checks') and t.get('prompt')
            identity=question_identity(b)
            if qid in selected:
                assert selected[qid][2]==identity,'Conflicting revisions require an explicit source selection'
                continue
            selected[qid]=(raw,b,identity)
    assert selected,'No question bundles'
    out=fresh(out);(out/'bundles').mkdir()
    rows=[]
    for qid,(raw,b,identity) in sorted(selected.items()):
        rel='bundles/'+qid+'.json';(out/rel).write_bytes(raw)
        rows.append({'id':qid,'path':rel,'bundleSha256':digest(raw),'questionSha256':identity,'family':b['task']['family'],'domain':b['task'].get('domain',b.get('review',{}).get('domain')),'questionReview':b.get('review',{}).get('semanticReview','unspecified')})
    manifest={'schemaVersion':1,'mode':'questions','questionSetId':digest(encoded([(r['id'],r['questionSha256'])for r in rows])),'questionCount':len(rows),'trajectoryCount':0,'trainingRows':0,'questions':rows}
    write(out/'manifest.json',manifest);return manifest

def load_questions(root):
    root=Path(root);m=read(root/'manifest.json');assert m['mode']=='questions'
    for r in m['questions']:
        raw=(root/r['path']).read_bytes()
        assert digest(raw)==r['bundleSha256'] and question_identity(json.loads(raw))==r['questionSha256'],'Question snapshot changed'
    assert m['questionSetId']==digest(encoded([(r['id'],r['questionSha256'])for r in m['questions']]))
    return m

def plan_trajectories(questions,profiles,out):
    q=load_questions(questions);plans=[];seen=set()
    for profile in profiles:
        key,spec=profile_identity(profile);assert key not in seen,'Duplicate teacher profile';seen.add(key)
        plans.append({'profileId':key,'teacher':spec,'jobs':[{'questionId':r['id'],'questionSha256':r['questionSha256'],'jobId':digest(encoded([r['questionSha256'],key])),'state':'pending'}for r in q['questions']]})
    assert plans
    out=fresh(out)
    result={'schemaVersion':1,'mode':'trajectories','state':'planned','questionsRoot':str(Path(questions).resolve()),'questionSetId':q['questionSetId'],'questionCount':q['questionCount'],'teachers':plans,'trainingRows':0,'authorizedToStart':False}
    write(out/'plan.json',result);return result

def export_training(questions,entries,out):
    """entries: profile + run + review + packedDataset per completed attempt.

    Packed files must come from the existing native XML renderer/tokenizer exporter.
    This function selects exact source rows; it never invents or edits a teacher answer.
    """
    qm=load_questions(questions);qmap={r['id']:r for r in qm['questions']};prepared=[];identities=set()
    for entry in entries:
        profile=read(entry['profile']);pk,ps=profile_identity(profile)
        raw=Path(entry['run']).read_bytes();run=json.loads(raw);review=read(entry['review']);qid=run['task']['id'];q=qmap[qid]
        bundle=read(Path(questions)/q['path']);expected=question_identity({**bundle,'task':run['task']})
        assert expected==q['questionSha256'],'Run used another question revision'
        assert run.get('teacherProfileId')==pk and run.get('questionSha256')==q['questionSha256'],'Collection profile identity missing or changed'
        assert run['model']=={'provider':ps['provider'],'id':ps['model']},'Teacher mismatch'
        assert run['role']=='teacher' and run['status']=='passed' and run['checks'] and all(c['passed']for c in run['checks'])
        generation=run.get('generation',{})
        for k,gk in [('thinking','thinkingLevel'),('contextWindow','contextWindow'),('maxTokens','maxTokens')]:
            if k in ps:assert generation.get(gk)==ps[k],'Generation profile mismatch: '+k
        for field in ('appendSystemPrompt','promptSuffix'):
            if field in ps:assert run.get(field)==ps[field],'Prompt profile mismatch: '+field
        assistants=[m for m in run['messages']if m['role']=='assistant']
        assert assistants and all(m.get('provider')==ps['provider'] and m.get('model')==ps['model']for m in assistants),'Mixed teacher messages'
        if bundle.get('followups'):
            turns=run.get('conversation',{}).get('turns',[])
            assert run.get('conversation',{}).get('allTurnsCompleted') and [t['userPrompt']for t in turns]==[bundle['task']['prompt']]+[t['prompt']for t in bundle['followups']],'Follow-up mismatch'
        if bundle['task'].get('domain')=='agent-orchestration':
            assert run.get('collectionMode')=='real-teacher-rwkv-delegation-v1' and run.get('delegationAcceptance',{}).get('passed'),'Real RWKV delegation required'
        assert review.get('verdict')=='pass' and review.get('reviewer') and review.get('notes')
        assert review.get('runSha256')==digest(raw) and review.get('questionSha256')==q['questionSha256'] and review.get('profileId')==pk,'Stale review'
        assert type(review.get('effectiveToolCalls'))is int and review['effectiveToolCalls']>=20
        assert all(review.get(k)is True for k in ['semanticPassed','wirePassed','scopePassed','noPadding']),'Incomplete trajectory review'
        packed=Path(entry['packedDataset']);pm=read(packed/'manifest.json')
        assert pm['format']=='rwkv-lightning-segments-eos-target-v2' and pm['requiredTrainingCapability']=='segment-end-target-v1'
        assert pm.get('toolFormat')=='xml' and pm.get('promptVersion')=='rwkv-pi-xml-continuation-v3'
        assert any(s['runId']==run['runId'] and s['sha256']==digest(raw)for s in pm['sources']),'Packed source mismatch'
        data=(packed/'train.jsonl').read_bytes();assert digest(data)==pm['files']['train.jsonl']
        lines=data.splitlines(keepends=True);rows=[r for r in pm['rows']if r['split']=='train'];assert len(lines)==len(rows)
        chosen=[(r,l)for r,l in zip(rows,lines)if r['runId']==run['runId']];assert chosen
        for _,line in chosen:
            segments=json.loads(line)['segments'];assert segments and any(s.get('train')for s in segments)
            assert all(s.get('end_target')is not None for s in segments if s.get('train'))
        assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}',run['runId']),'Unsafe run ID'
        key=(pk,run['runId']);assert key not in identities,'Duplicate run';identities.add(key)
        prepared.append((pk,ps,q,run,raw,review,chosen,pm))
    assert prepared,'No reviewed training rows'
    out=fresh(out);shutil.copytree(questions,out/'questions')
    result={'schemaVersion':1,'mode':'trajectories','state':'reviewed_training_export','questionSetId':qm['questionSetId'],'questionCount':len({x[2]['id']for x in prepared}),'trajectoryCount':len(prepared),'trainingRows':0,'teachers':[]}
    for pk in sorted({x[0]for x in prepared}):
        items=[x for x in prepared if x[0]==pk];folder=out/'teachers'/pk[:16];folder.mkdir(parents=True);(folder/'runs').mkdir();(folder/'reviews').mkdir()
        # Tokenizer and engine identity cannot silently change within one training file.
        signatures={tuple(x[7].get(k)for k in ['engineRevision','vocabSha256','promptVersion','requiredTrainingCapability'])for x in items};assert len(signatures)==1
        index=[]
        with (folder/'train.jsonl').open('wb')as stream:
            for _,ps,q,run,raw,review,chosen,pm in items:
                (folder/'runs'/f"{run['runId']}.json").write_bytes(raw);write(folder/'reviews'/f"{run['runId']}.json",review)
                for row,line in chosen:stream.write(line);index.append({**row,'questionId':q['id'],'questionSha256':q['questionSha256'],'profileId':pk,'sourceRunSha256':digest(raw)})
        teacher={'profileId':pk,'teacher':items[0][1],'questionCount':len({x[2]['id']for x in items}),'trajectoryCount':len(items),'trainingRows':len(index),'counts':{'train':len(index),'validation':0,'test':0},'engineRevision':items[0][7]['engineRevision'],'vocabSha256':items[0][7]['vocabSha256'],'promptVersion':items[0][7]['promptVersion'],'requiredTrainingCapability':items[0][7]['requiredTrainingCapability'],'toolFormat':'xml','sources':[{'runId':x[3]['runId'],'sha256':digest(x[4])}for x in items],'format':'rwkv-lightning-segments-eos-target-v2','protocol':'canonical XML+JSON','rows':index,'files':{'train.jsonl':digest((folder/'train.jsonl').read_bytes())}}
        teacher.update(schemaVersion=2,tokenizerProof=items[0][7].get('tokenizerProof'),maxTokens=max(r.get('tokens',0)for r in index),requiredCtx=max(r.get('tokens',0)for r in index),verification='Exact rows selected from reviewed native packs; teacher profiles remain separate')
        for split in ['validation','test']:
            (folder/(split+'.jsonl')).write_bytes(b'');teacher['files'][split+'.jsonl']=digest(b'')
        write(folder/'manifest.json',teacher);result['teachers'].append({'path':str(folder.relative_to(out)),**{k:teacher[k]for k in ['profileId','teacher','questionCount','trajectoryCount','trainingRows']}});result['trainingRows']+=len(index)
    write(out/'manifest.json',result);return result

def compare_attempts(attempts):
    """Compare only identical question revisions. Attempts retain their own model/config."""
    groups=collections.defaultdict(list)
    for row in attempts:groups[(row['questionId'],row['questionSha256'])].append(row)
    return {'questionVersions':len(groups),'questions':[{'questionId':qid,'questionSha256':qh,'attempts':rows}for (qid,qh),rows in sorted(groups.items())]}
