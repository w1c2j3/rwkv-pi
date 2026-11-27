// Native-Pi adapter for independently configured OpenAI-compatible teacher endpoints.
// Budget module, account lock, credentials and runtime are supplied by the private profile.
import {readFileSync,writeFileSync,mkdirSync,copyFileSync,existsSync,rmSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash,randomUUID} from 'node:crypto';
import {createProvider} from '@earendil-works/pi-ai';
import {openAICompletionsApi} from '@earendil-works/pi-ai/api/openai-completions.lazy';
import {openAIResponsesApi} from '@earendil-works/pi-ai/api/openai-responses.lazy';
import {createProductionWebTools} from './delegation.mjs';
process.setMaxListeners(0);
process.on('uncaughtException',err=>{console.error('[collect uncaughtException]',err?.code??'',err?.message??err);});
process.on('unhandledRejection',err=>{console.error('[collect unhandledRejection]',err?.code??'',err?.message??err);});
const [root,profilePath,profileId]=process.argv.slice(2);
const read=p=>JSON.parse(readFileSync(p,'utf8'));
const sha=b=>createHash('sha256').update(b).digest('hex');
const stable=x=>Array.isArray(x)?x.map(stable):x&&typeof x==='object'?Object.fromEntries(Object.keys(x).sort().map(k=>[k,stable(x[k])])):x;
const profile=read(profilePath),plan=read(join(root,'plan.json')),teacher=plan.teachers.find(x=>x.profileId===profileId);
// Python verifies the complete profile hash before starting; compare values here too.
if(!teacher)throw Error('Frozen profile identity missing');
for(const [key,value]of Object.entries(teacher.teacher))if(JSON.stringify(stable(profile[key]))!==JSON.stringify(stable(value)))throw Error('Teacher profile changed: '+key);
if(!['openai-completions','openai-responses'].includes(profile.api))throw Error('This adapter requires an OpenAI-compatible Chat Completions or Responses endpoint');
const ex=profile.execution,bp=profile.budget;
const {PaidTrialBudget,atomicJson}=await import(pathToFileURL(ex.budgetModule).href);
const {runTask}=await import(pathToFileURL(ex.harnessModule).href);
const target=join(root,'teachers',profileId.slice(0,16));mkdirSync(target,{recursive:true});
const resume=process.env.RWKV_RESUME_COLLECTION==='1'&&existsSync(join(target,'experiment.json'));
if(existsSync(join(target,'experiment.json'))&&!resume)throw Error('Existing attempt is never replayed; create a new plan for a deliberate comparison');
for(const name of ['raw','traces','runs'])mkdirSync(join(target,name),{recursive:true});
const pricing=existsSync(bp.ledgerPath)?read(bp.ledgerPath).pricing:bp.pricing;
const rate=pricing?.[profile.model];
if(!rate||rate.provider!==profile.provider||rate.baseUrl!==profile.baseUrl||![rate.input,rate.output].every(x=>Number.isFinite(x)&&x>0)||(rate.cachedInput!==undefined&&(!Number.isFinite(rate.cachedInput)||rate.cachedInput<0)))throw Error('Verified prices for this teacher are required in the account ledger');
const budget=new PaidTrialBudget({path:bp.ledgerPath,budgetCny:bp.cnyLimit,arms:pricing,caps:{[profile.provider]:ex.apiConcurrency??2},productionAuthorization:bp.productionAuthorization,rawDirectory:join(target,'raw'),rpm:ex.rpm??120,tpm:ex.tpm??1000000});
const before=budget.charged(),start=budget.state.requests.length;
budget.admissionLimitMicroCny=Math.min(Math.floor(bp.cnyLimit*1e6),before+Math.floor(bp.incrementCny*1e6));
const questions=read(join(plan.questionsRoot,'manifest.json'));
if(questions.questionSetId!==plan.questionSetId)throw Error('Question set changed');
if(!Number.isFinite(bp.incrementCny)||bp.incrementCny<=0)throw Error('Positive incremental budget required');
const prev=resume?read(join(target,'experiment.json')):null;
const passedRows=prev?(prev.results??[]).filter(r=>r.status==='passed'||r.status==='student_passed'):[];
const passedIds=new Set(passedRows.map(r=>r.questionId));
const state={teacher:teacher.teacher,profileId,questionSetId:plan.questionSetId,startedAt:prev?.startedAt??new Date().toISOString(),results:[...passedRows],trainingRows:0};
const save=()=>atomicJson(join(target,'experiment.json'),state);save();
const key=readFileSync(profile.credentialFile,'utf8').trim();
function providerFor(qid,taskKey=qid){
 const trace=join(target,'traces',qid);rmSync(trace,{recursive:true,force:true});mkdirSync(trace,{recursive:true});let sequence=0;
 const sessionId=`${profileId.slice(0,12)}-${taskKey}-${randomUUID().slice(0,8)}`;
 const headers={...(profile.headers??{}),'x-litellm-session-id':sessionId};
 const model={id:profile.model,name:profile.model,provider:profile.provider,api:profile.api,baseUrl:profile.baseUrl,reasoning:profile.thinking!=='off',input:['text'],contextWindow:profile.contextWindow,maxTokens:profile.maxTokens,cost:{input:0,output:0,cacheRead:0,cacheWrite:0},headers,samplingParams:profile.samplingParams??{},compat:profile.compat??{supportsStore:false,supportsDeveloperRole:false,maxTokensField:'max_tokens'}};
 const apiImpl=profile.api==='openai-responses'?openAIResponsesApi():openAICompletionsApi();
 const provider=createProvider({id:profile.provider,models:[model],baseUrl:profile.baseUrl,api:apiImpl,auth:{apiKey:{name:'Configured teacher',resolve:async()=>({auth:{apiKey:key,headers}})}}});
 const fetcher=(u,i)=>{
  const n=String(++sequence).padStart(3,'0'),body=JSON.parse(i.body);
  const maxOut=body.max_completion_tokens??body.max_tokens??body.max_output_tokens;
  if(body.model!==profile.model||maxOut!==profile.maxTokens)throw Error('Unexpected teacher request');
  atomicJson(join(trace,n+'.request.json'),{sessionId,...body});
  const mergedHeaders=new Headers(i.headers);
  mergedHeaders.set('x-litellm-session-id',sessionId);
  if(!mergedHeaders.has('authorization'))mergedHeaders.set('authorization','Bearer '+key);
  return budget.forArm(profile.model,profileId+'/'+taskKey+'/'+n)(u,{...i,headers:mergedHeaders});
 };
 return {...provider,rwkvPiEditMode:profile.editMode??'line',stream:(m,c,o)=>provider.stream(m,c,{...o,maxRetries:0,fetch:fetcher}),streamSimple:(m,c,o)=>provider.streamSimple(m,c,{...o,maxRetries:0,fetch:fetcher})};
}
const queue=teacher.jobs.filter(j=>!passedIds.has(j.questionId));
function searchCorpus(corpus,query,limit=5){
 const terms=String(query??'').toLowerCase().match(/[\u4e00-\u9fff]{1,4}|[a-z0-9_.-]{2,}/g)??[];
 const scored=corpus.map((p,idx)=>{
  const hay=`${p.title}\n${p.url}\n${p.snippet}\n${p.content}`.toLowerCase();
  const titleHay=`${p.title}\n${p.url}`.toLowerCase();
  let score=0;
  for(const t of terms){if(titleHay.includes(t))score+=3;else if(hay.includes(t))score+=1;}
  return {p,score,idx};
 }).sort((a,b)=>b.score-a.score||a.idx-b.idx);
 return {results:scored.slice(0,Math.max(1,Math.min(5,limit))).map(({p})=>({title:p.title,url:p.url,date:p.date,snippet:p.snippet,summary:p.snippet}))};
}
const maxAttempts=profile.rejectionSampling?.enabled?Math.max(1,profile.rejectionSampling.maxAttemptsPerQuestion??1):1;
let daggerStudentProvider,activeStudents=0;const studentWaiters=[];
const maxStudentSlots=ex.studentConcurrency??32;
const acquireStudentSlot=()=>activeStudents<maxStudentSlots?(activeStudents++,Promise.resolve()):new Promise(r=>studentWaiters.push(r));
const releaseStudentSlot=()=>{const next=studentWaiters.shift();if(next)next();else activeStudents--;};
async function getDaggerStudentProvider(){
 if(daggerStudentProvider)return daggerStudentProvider;
 const student=read(ex.studentConfig);
 const {createRwkvProvider,readConfig}=await import(pathToFileURL(join(ex.runtime,'dist','provider.js')).href);
 const cfg=readConfig({...process.env,RWKV_BACKEND:student.backend??'lightning',RWKV_BASE_URL:student.baseUrl,RWKV_MODEL:student.modelId,RWKV_API_KEY:student.apiKey??'local-rwkv',RWKV_CONTEXT_WINDOW:String(student.contextWindow??32768),RWKV_MAX_TOKENS:String(student.maxTokens??6144),RWKV_TOOL_FORMAT:student.toolFormat??'xml',RWKV_PROMPT_TEMPLATE:student.promptTemplate??'xml',RWKV_TOOL_CALL_RECOVERY:'1',RWKV_IDENTICAL_CALL_LIMIT:'2',RWKV_EDIT_MODE:student.editMode??profile.editMode??'line',RWKV_HISTORY_MODE:student.historyMode??'epoch',RWKV_EPOCH_BUDGET_TOKENS:String(student.epochBudgetTokens??9600),RWKV_EPOCH_KEEP_RECENT_PAIRS:String(student.epochKeepRecentPairs??2),RWKV_EPOCH_MAX_RESULT_TOKENS:String(student.epochMaxResultTokens??2000),...(student.adapterId?{RWKV_ADAPTER_ID:String(student.adapterId)}:{}),...(student.adapterVersion?{RWKV_ADAPTER_VERSION:String(student.adapterVersion)}:{}),...(student.adapterScale!==undefined?{RWKV_ADAPTER_SCALE:String(student.adapterScale)}:{})});
 daggerStudentProvider=createRwkvProvider(cfg);return daggerStudentProvider;
}
async function one(job){
 const q=questions.questions.find(q=>q.id===job.questionId&&q.questionSha256===job.questionSha256);if(!q)throw Error('Question revision missing');
 const raw=readFileSync(join(plan.questionsRoot,q.path));if(sha(raw)!==q.bundleSha256)throw Error('Question bytes changed');
 const bundle=JSON.parse(raw);
 const corpus=Array.isArray(bundle.webCorpus)&&bundle.webCorpus.length?bundle.webCorpus:null;
 const wantsWeb=Boolean(corpus)||(bundle.task.tools??[]).some(t=>t==='web_search'||t==='web_fetch')||Boolean(profile.webTools?.enabled);
 const webKey=wantsWeb&&!corpus&&profile.webTools?.credentialFile?readFileSync(profile.webTools.credentialFile,'utf8').trim():key;
 const begin=Date.now();let row;
 for(let attempt=1;attempt<=maxAttempts&&!budget.state.halted;attempt++){
  const taskKey=attempt===1?q.id:`${q.id}@${attempt}`;
  const directSpillDir=wantsWeb?join(target,'runs',`spill-${taskKey}-${randomUUID().slice(0,8)}`):undefined;
  const makeWebTools=(phase='teacher',ctx={})=>{
   const spillHostDir=ctx.spillHostDir??directSpillDir;
   return createProductionWebTools({maxCalls:profile.webTools?.maxCalls??32,onSpill:spillHostDir?(url,content)=>{
    mkdirSync(spillHostDir,{recursive:true,mode:0o700});
    const name=`dsh-spill-${createHash('sha256').update(url).digest('hex').slice(0,12)}.txt`;
    writeFileSync(join(spillHostDir,name),content,{mode:0o600});
    return `/tmp/dsh-spill/${name}`;
   }:undefined,callWeb:async(kind,body,idx,signal)=>{
    if(!corpus)return budget.callWebTool(kind,body,{taskId:`${profileId}/${taskKey}/${phase}-web-${String(idx).padStart(3,'0')}`,apiKey:webKey,baseUrl:profile.webTools?.baseUrl,signal});
    if(kind==='search')return searchCorpus(corpus,body.query,body.count??5);
    const hit=corpus.find(p=>p.url===body.url)||corpus.find(p=>p.url.replace(/\/+$/,'')===String(body.url??'').replace(/\/+$/,''));
    if(!hit)throw Error(`HTTP 404 Not Found: ${body.url}`);
    return {title:hit.title,content:hit.content};
   }});
  };
  const webTools=wantsWeb?makeWebTools('teacher',{spillHostDir:directSpillDir}):undefined;
  const options={provider:providerFor(q.id,taskKey),role:'teacher',outputDir:join(target,'runs'),...(directSpillDir?{spillHostDir:directSpillDir}:{}),followups:bundle.followups??[],maxRequests:ex.maxRequests??96,timeoutMs:ex.timeoutMs??1800000,sandbox:'bubblewrap',thinkingLevel:profile.thinking,editMode:profile.editMode??'line',appendSystemPrompt:profile.appendSystemPrompt,promptSuffix:profile.promptSuffix??'',...(webTools?{webTools}:{}),...(profile.historyMode?{historyMode:profile.historyMode,epoch:{...(profile.epochBudgetTokens?{budgetTokens:profile.epochBudgetTokens}:{}),...(profile.epochKeepRecentPairs?{keepRecentPairs:profile.epochKeepRecentPairs}:{}),...(profile.epochMaxResultTokens?{maxResultTokens:profile.epochMaxResultTokens}:{})}}:{})};
  try{
   let result;
   if(bundle.task.domain==='agent-orchestration'){
    if(!ex.delegationModule)throw Error('Real delegation adapter required; simulator fallback is forbidden');
    const {runDelegatedTask}=await import(pathToFileURL(ex.delegationModule).href);const student=read(ex.studentConfig);
    result=await runDelegatedTask(bundle.task,{...options,runtime:ex.runtime,teacherIdentity:{provider:profile.provider,id:profile.model},studentModel:student.modelId,studentEndpoint:student.baseUrl,childLimits:{concurrency:2,maxRequests:32,timeoutSeconds:900,maxTasks:6,sandbox:'bwrap',mount:'/srv/workspace'},immutableFiles:bundle.review?.immutableFiles});
   }else if(ex.daggerModule){
    const {runDaggerTask}=await import(pathToFileURL(ex.daggerModule).href);
    const studentProvider=await getDaggerStudentProvider();
    result=await runDaggerTask(bundle.task,{...options,runtime:ex.runtime,studentProvider,acquireStudentSlot,releaseStudentSlot,studentMaxRequests:ex.studentMaxRequests??20,studentTimeoutMs:ex.studentTimeoutMs??480000,createWebTools:wantsWeb?makeWebTools:undefined,immutableFiles:bundle.review?.immutableFiles});
   }else result=await runTask(bundle.task,options);
   const {record,path}=result;record.teacherProfileId=profileId;record.questionSha256=q.questionSha256;record.trainingEligible=false;atomicJson(path,record);
   const assistant=record.messages.filter(m=>m.role==='assistant'),calls=assistant.flatMap(m=>m.content.filter(b=>b.type==='toolCall'));
   const rowStatus=result.kind==='student_passed'?'student_passed':record.status;
   row={questionId:q.id,questionSha256:q.questionSha256,profileId,task:q.id,taskKey,attempts:attempt,runId:record.runId,run:path,runSha256:sha(readFileSync(path)),status:rowStatus,rawToolCalls:calls.length,effectiveToolCalls:null,userTurns:record.conversation?.turns.length??1,assistantDecisionRounds:assistant.length,trainingEligible:false,...(result.studentFailureKind?{studentFailureKind:result.studentFailureKind}:{}),...(record.dagger?{studentSteps:record.dagger.studentPrefixSteps,teacherSteps:record.dagger.teacherRecoverySteps}:{})};
   if(rowStatus==='passed'||rowStatus==='student_passed')break;
  }catch(error){row={questionId:q.id,questionSha256:q.questionSha256,profileId,task:q.id,taskKey,attempts:attempt,status:'harness_error',error:String(error.message),trainingEligible:false};}
 }
 const tk=row.taskKey??row.questionId;
 const reqs=budget.state.requests.slice(start).filter(r=>r.taskId.startsWith(profileId+'/'+tk+'/'));
 for(const r of reqs){
  const [,,n]=r.taskId.split('/');
  const rawSse=join(target,'raw',r.id+'.sse');if(existsSync(rawSse))copyFileSync(rawSse,join(target,'traces',q.id,n+'.sse'));
  const rawWeb=join(target,'raw',r.id+'.web.json');if(existsSync(rawWeb))copyFileSync(rawWeb,join(target,'traces',q.id,n+'.json'));
 }
 const modelReqs=reqs.filter(r=>!r.kind),webReqs=reqs.filter(r=>Boolean(r.kind));
 row.requestCount=modelReqs.length;if(webReqs.length)row.webCalls=webReqs.length;
 row.costUpperCny=reqs.reduce((n,r)=>n+(r.chargedMicroCny??r.reservedMicroCny),0)/1e6;
 row.usageComplete=modelReqs.every(r=>r.usage);
 row.reportedTokens=modelReqs.reduce((n,r)=>({input:n.input+(r.usage?.prompt_tokens??0),output:n.output+(r.usage?.completion_tokens??0)}),{input:0,output:0});
 row.seconds=(Date.now()-begin)/1000;state.results.push(row);save();
}
try{await Promise.all(Array.from({length:ex.workflowConcurrency??2},async()=>{while(queue.length&&!budget.state.halted)await one(queue.shift());}));}
finally{
 await budget.settled();
 const chosenKeys=new Map(state.results.map(r=>[r.taskKey??r.questionId,r.questionId]));
 for(const r of budget.state.requests.slice(start)){
  const [pk,tk,n]=r.taskId.split('/'),qid=chosenKeys.get(tk);
  if(pk!==profileId||!qid)continue;
  const rawSse=join(target,'raw',r.id+'.sse');if(existsSync(rawSse))copyFileSync(rawSse,join(target,'traces',qid,n+'.sse'));
  const rawWeb=join(target,'raw',r.id+'.web.json');if(existsSync(rawWeb))copyFileSync(rawWeb,join(target,'traces',qid,n+'.json'));
 }
 for(const row of state.results){
  const tk=row.taskKey??row.questionId;
  const requests=budget.state.requests.slice(start).filter(r=>r.taskId.startsWith(profileId+'/'+tk+'/'));
  const modelRequests=requests.filter(r=>!r.kind),webRequests=requests.filter(r=>Boolean(r.kind));
  row.requestCount=modelRequests.length;if(webRequests.length)row.webCalls=webRequests.length;
  row.costUpperCny=requests.reduce((n,r)=>n+(r.chargedMicroCny??r.reservedMicroCny),0)/1e6;
  row.usageComplete=modelRequests.every(r=>r.usage);
  row.reportedTokens=modelRequests.reduce((n,r)=>({input:n.input+(r.usage?.prompt_tokens??0),output:n.output+(r.usage?.completion_tokens??0)}),{input:0,output:0});
 }
 state.finishedAt=new Date().toISOString();state.costUpperCny=(budget.charged()-before)/1e6;state.budgetHalted=budget.state.halted;state.pendingJobs=queue;
 state.state=queue.length||budget.state.halted||state.results.some(r=>r.status==='harness_error')?'collection_stopped':'awaiting_independent_trajectory_review';save();
 if(queue.length||budget.state.halted||state.results.some(r=>r.status==='harness_error'))process.exitCode=1;
}
