// Native-Pi adapter for independently configured OpenAI-compatible teacher endpoints.
// Budget module, account lock, credentials and runtime are supplied by the private profile.
import {readFileSync,writeFileSync,mkdirSync,copyFileSync,existsSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {createProvider} from '@earendil-works/pi-ai';
import {openAICompletionsApi} from '@earendil-works/pi-ai/api/openai-completions.lazy';
const [root,profilePath,profileId]=process.argv.slice(2);
const read=p=>JSON.parse(readFileSync(p,'utf8'));
const sha=b=>createHash('sha256').update(b).digest('hex');
const stable=x=>Array.isArray(x)?x.map(stable):x&&typeof x==='object'?Object.fromEntries(Object.keys(x).sort().map(k=>[k,stable(x[k])])):x;
const profile=read(profilePath),plan=read(join(root,'plan.json')),teacher=plan.teachers.find(x=>x.profileId===profileId);
// Python verifies the complete profile hash before starting; compare values here too.
if(!teacher)throw Error('Frozen profile identity missing');
for(const [key,value]of Object.entries(teacher.teacher))if(JSON.stringify(stable(profile[key]))!==JSON.stringify(stable(value)))throw Error('Teacher profile changed: '+key);
if(profile.api!=='openai-completions')throw Error('This adapter requires an OpenAI-compatible Chat Completions endpoint');
const ex=profile.execution,bp=profile.budget;
const {PaidTrialBudget,atomicJson}=await import(pathToFileURL(ex.budgetModule).href);
const {runTask}=await import(pathToFileURL(ex.harnessModule).href);
const target=join(root,'teachers',profileId.slice(0,16));mkdirSync(target,{recursive:true});
if(existsSync(join(target,'experiment.json')))throw Error('Existing attempt is never replayed; create a new plan for a deliberate comparison');
for(const name of ['raw','traces','runs'])mkdirSync(join(target,name));
const pricing=existsSync(bp.ledgerPath)?read(bp.ledgerPath).pricing:bp.pricing;
const rate=pricing?.[profile.model];
if(!rate||rate.provider!==profile.provider||rate.baseUrl!==profile.baseUrl||![rate.input,rate.output].every(x=>Number.isFinite(x)&&x>0)||(rate.cachedInput!==undefined&&(!Number.isFinite(rate.cachedInput)||rate.cachedInput<0)))throw Error('Verified prices for this teacher are required in the account ledger');
const budget=new PaidTrialBudget({path:bp.ledgerPath,budgetCny:bp.cnyLimit,arms:pricing,caps:{[profile.provider]:ex.apiConcurrency??2},productionAuthorization:bp.productionAuthorization,rawDirectory:join(target,'raw'),rpm:ex.rpm??120,tpm:ex.tpm??1000000});
const before=budget.charged(),start=budget.state.requests.length;
budget.admissionLimitMicroCny=Math.min(Math.floor(bp.cnyLimit*1e6),before+Math.floor(bp.incrementCny*1e6));
const questions=read(join(plan.questionsRoot,'manifest.json'));
if(questions.questionSetId!==plan.questionSetId)throw Error('Question set changed');
if(!Number.isFinite(bp.incrementCny)||bp.incrementCny<=0)throw Error('Positive incremental budget required');
const state={teacher:teacher.teacher,profileId,questionSetId:plan.questionSetId,startedAt:new Date().toISOString(),results:[],trainingRows:0};
const save=()=>atomicJson(join(target,'experiment.json'),state);save();
const key=readFileSync(profile.credentialFile,'utf8').trim();
function providerFor(qid){
 const trace=join(target,'traces',qid);mkdirSync(trace);let sequence=0;
 const model={id:profile.model,name:profile.model,provider:profile.provider,api:profile.api,baseUrl:profile.baseUrl,reasoning:profile.thinking!=='off',input:['text'],contextWindow:profile.contextWindow,maxTokens:profile.maxTokens,cost:{input:0,output:0,cacheRead:0,cacheWrite:0},samplingParams:profile.samplingParams??{},compat:profile.compat??{supportsStore:false,supportsDeveloperRole:false,maxTokensField:'max_tokens'}};
 const provider=createProvider({id:profile.provider,models:[model],baseUrl:profile.baseUrl,api:openAICompletionsApi(),auth:{apiKey:{name:'Configured teacher',resolve:async()=>({auth:{apiKey:key}})}}});
 const fetcher=(u,i)=>{
  const n=String(++sequence).padStart(3,'0'),body=JSON.parse(i.body);
  if(body.model!==profile.model||body.max_tokens!==profile.maxTokens)throw Error('Unexpected teacher request');
  atomicJson(join(trace,n+'.request.json'),body);
  return budget.forArm(profile.model,profileId+'/'+qid+'/'+n)(u,i);
 };
 return {...provider,stream:(m,c,o)=>provider.stream(m,c,{...o,maxRetries:0,fetch:fetcher}),streamSimple:(m,c,o)=>provider.streamSimple(m,c,{...o,maxRetries:0,fetch:fetcher})};
}
const queue=[...teacher.jobs];
async function one(job){
 const q=questions.questions.find(q=>q.id===job.questionId&&q.questionSha256===job.questionSha256);if(!q)throw Error('Question revision missing');
 const raw=readFileSync(join(plan.questionsRoot,q.path));if(sha(raw)!==q.bundleSha256)throw Error('Question bytes changed');
 const bundle=JSON.parse(raw),options={provider:providerFor(q.id),role:'teacher',outputDir:join(target,'runs'),followups:bundle.followups??[],maxRequests:ex.maxRequests??96,timeoutMs:ex.timeoutMs??1800000,sandbox:'bubblewrap',thinkingLevel:profile.thinking,appendSystemPrompt:profile.appendSystemPrompt,promptSuffix:profile.promptSuffix??''};
 const begin=Date.now();let row;
 try{
  let result;
  if(bundle.task.domain==='agent-orchestration'){
   if(!ex.delegationModule)throw Error('Real delegation adapter required; simulator fallback is forbidden');
   const {runDelegatedTask}=await import(pathToFileURL(ex.delegationModule).href);const student=read(ex.studentConfig);
   result=await runDelegatedTask(bundle.task,{...options,runtime:ex.runtime,teacherIdentity:{provider:profile.provider,id:profile.model},studentModel:student.modelId,studentEndpoint:student.baseUrl,childLimits:{concurrency:2,maxRequests:32,timeoutSeconds:900,maxTasks:6,sandbox:'bwrap',mount:'/srv/workspace'},immutableFiles:bundle.review.immutableFiles});
  }else result=await runTask(bundle.task,options);
  const {record,path}=result;record.teacherProfileId=profileId;record.questionSha256=q.questionSha256;record.trainingEligible=false;atomicJson(path,record);
  const assistant=record.messages.filter(m=>m.role==='assistant'),calls=assistant.flatMap(m=>m.content.filter(b=>b.type==='toolCall'));
  row={questionId:q.id,questionSha256:q.questionSha256,profileId,task:q.id,runId:record.runId,run:path,runSha256:sha(readFileSync(path)),status:record.status,rawToolCalls:calls.length,effectiveToolCalls:null,userTurns:record.conversation?.turns.length??1,assistantDecisionRounds:assistant.length,trainingEligible:false};
 }catch(error){row={questionId:q.id,questionSha256:q.questionSha256,profileId,task:q.id,status:'harness_error',error:String(error.message),trainingEligible:false};}
 row.seconds=(Date.now()-begin)/1000;state.results.push(row);save();
}
try{await Promise.all(Array.from({length:ex.workflowConcurrency??2},async()=>{while(queue.length&&!budget.state.halted)await one(queue.shift());}));}
finally{
 await budget.settled();
 for(const r of budget.state.requests.slice(start)){
  const [pk,qid,n]=r.taskId.split('/');const raw=join(target,'raw',r.id+'.sse');if(pk===profileId&&existsSync(raw))copyFileSync(raw,join(target,'traces',qid,n+'.sse'));
 }
 for(const row of state.results){
  const requests=budget.state.requests.slice(start).filter(r=>r.taskId.startsWith(profileId+'/'+row.questionId+'/'));
  row.requestCount=requests.length;row.costUpperCny=requests.reduce((n,r)=>n+(r.chargedMicroCny??r.reservedMicroCny),0)/1e6;
  row.usageComplete=requests.every(r=>r.usage);
  row.reportedTokens=requests.reduce((n,r)=>({input:n.input+(r.usage?.prompt_tokens??0),output:n.output+(r.usage?.completion_tokens??0)}),{input:0,output:0});
 }
 state.finishedAt=new Date().toISOString();state.costUpperCny=(budget.charged()-before)/1e6;state.budgetHalted=budget.state.halted;state.pendingJobs=queue;
 state.state=queue.length||budget.state.halted||state.results.some(r=>r.status==='harness_error')?'collection_stopped':'awaiting_independent_trajectory_review';save();
 if(queue.length||budget.state.halted||state.results.some(r=>r.status==='harness_error'))process.exitCode=1;
}
