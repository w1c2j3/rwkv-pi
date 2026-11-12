// Real teacher -> RWKV collection. Runtime and task assets remain outside this repository.
import {mkdirSync, writeFileSync, readFileSync, appendFileSync, existsSync} from 'node:fs';
import {join, resolve, dirname} from 'node:path';
import {pathToFileURL} from 'node:url';
import {randomUUID, createHash} from 'node:crypto';
import {InMemoryCredentialStore, InMemoryModelsStore} from '@earendil-works/pi-ai';
import {createAgentSession, createExtensionRuntime, ModelRuntime, SessionManager, SettingsManager} from '@earendil-works/pi-coding-agent';
import {delegationTools, delegationEligibility} from './delegation.mjs';

const save = (p, x) => writeFileSync(p, JSON.stringify(x, null, 2) + '\n', {mode:0o600});
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const WORKFLOW = '本次采用真实协作：你负责规划、委派、审查差异和核验；RWKV子agent自行阅读项目、运行命令、修改实现和验证。先给中文目标、必要上下文、可修改边界与验收要求，再调用subagent_plan和subagent_start。不要给预写补丁或逐行答案。独立工作可并行；有依赖的工作须等待结果审查应用。用subagent_wait收取结果，审查subagent_diff后才subagent_apply，最后subagent_verify独立验证。失败必须如实处理和说明；不能把未完成的工作说成成功。子agent只有read/write/edit/bash/grep/find/ls工具，命令工具是bash。凡测试、初始化、生成报告或修改代码均需write=true，false只用于静态阅读。委派任务要有实质业务目标；不要反复退化成pwd探针。根据返回的实际工具错误调整安排。所有分析与最终回复使用简体中文。';

export async function runDelegatedTask(input, options) {
  const load = name => import(pathToFileURL(join(options.runtime, 'dist', name + '.js')).href);
  const [{parseTask,taskPath},{digest,runtimeProvenance},{WorkspaceSandbox},{SubagentManager},{SubagentPlan}]
    = await Promise.all(['tasks','provenance','sandbox','subagents','subagent-plan'].map(load));
  const task=parseTask(input);
  if(task.environment)throw Error('This collector currently accepts the reviewed Linux sandbox tasks only');
  const runId=randomUUID(), dir=resolve(options.outputDir,runId), cwd=join(dir,'workspace');
  mkdirSync(cwd,{recursive:true});
  for(const [name,content] of Object.entries(task.files)) {
    const p=taskPath(cwd,name);mkdirSync(dirname(p),{recursive:true});writeFileSync(p,content,{flag:'wx'});
  }
  const model=options.provider.getModels()[0];
  if(model.provider!==options.teacherIdentity?.provider||model.id!==options.teacherIdentity?.id)throw Error('Configured teacher identity mismatch');
  if(options.followups?.length)throw Error('Delegation collector does not yet support user follow-ups');
  const sandbox=new WorkspaceSandbox(cwd);await sandbox.probe();await sandbox.probeSearch(['read','grep','find','ls']);
  const {id:_id,split:_split,...definition}=task;
  const record={schemaVersion:1,runId,task,taskSha256:digest(definition),role:'teacher',
    model:{provider:model.provider,id:model.id},provenance:runtimeProvenance(),isolation:sandbox.kind,responseAdapter:'none',
    startedAt:new Date().toISOString(),finishedAt:'',budget:{maxRequests:options.maxRequests,timeoutMs:options.timeoutMs},
    generation:{contextWindow:model.contextWindow,maxTokens:model.maxTokens,thinkingLevel:options.thinkingLevel},
    status:'error',checks:[],steps:[],messages:[],promptSuffix:options.promptSuffix,appendSystemPrompt:options.appendSystemPrompt,
    collectionMode:'real-teacher-rwkv-delegation-v1',supervision:{teacher:model.id,provider:model.provider,workerOutputs:'observations_only'},
    delegatedWorkers:[],delegationEvents:[],verificationRuns:[]};
  const checkpoint=()=>save(join(dir,'checkpoint.json'),record);
  const event=x=>appendFileSync(join(dir,'events.jsonl'),JSON.stringify(x)+'\n',{mode:0o600});
  const immutable=Object.fromEntries((options.immutableFiles??[]).map(name=>[name,sha(readFileSync(taskPath(cwd,name)))]));
  async function verify() {
    const checks=[];
    for(const check of task.checks) {
      try {
        if(check.type==='file_equals') {
          const actual=readFileSync(taskPath(cwd,check.path),'utf8');checks.push({passed:actual===check.expected,detail:'file_equals:'+check.path});
        } else {
          const r=await sandbox.exec(check.argv,{timeoutMs:check.timeoutMs});
          checks.push({passed:r.exitCode===0&&(check.expectedStdout===undefined||r.stdout.toString()===check.expectedStdout),
            detail:(r.stdout.toString()+r.stderr.toString()).slice(0,32768)});
        }
      }catch(e){checks.push({passed:false,detail:String(e.message)});}
    }
    const immutableChanges=Object.entries(immutable).filter(([name,h])=>{
      const p=taskPath(cwd,name);return !existsSync(p)||sha(readFileSync(p))!==h;
    }).map(([name])=>name);
    const result={at:Date.now(),passed:checks.every(c=>c.passed)&&immutableChanges.length===0,checks,immutableChanges};
    record.verificationRuns.push(result);checkpoint();return result;
  }
  const manager=new SubagentManager(cwd,options.childLimits,join(dir,'children'),a=>{
    event({type:'worker_state',id:a.id,state:a.state,requests:a.requests,toolCalls:a.toolCalls});
  });
  record.workerRuntime={model:options.studentModel,endpoint:options.studentEndpoint,limits:options.childLimits,innerWorkspace:manager.inner};
  const controlTools=delegationTools({manager,plan:new SubagentPlan(),verify,recordEvent:e=>{
    record.delegationEvents.push(e);event({type:'delegation',...e});checkpoint();
  }});
  const readTools=sandbox.tools(['read','grep','find','ls']);
  const resourceLoader={getExtensions:()=>({extensions:[],errors:[],runtime:createExtensionRuntime()}),getSkills:()=>({skills:[],diagnostics:[]}),
    getPrompts:()=>({prompts:[],diagnostics:[]}),getThemes:()=>({themes:[],diagnostics:[]}),getAgentsFiles:()=>({agentsFiles:[]}),
    getSystemPrompt:()=>undefined,getSystemPromptSource:()=>undefined,getAppendSystemPrompt:()=>[options.appendSystemPrompt,WORKFLOW],
    getAppendSystemPromptSources:()=>[],extendResources:()=>{},reload:async()=>{}};
  const modelRuntime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsStore:new InMemoryModelsStore(),modelsPath:null,refreshOnCreate:false,allowModelNetwork:false});
  modelRuntime.registerNativeProvider(options.provider);
  const {session}=await createAgentSession({cwd,agentDir:join(dir,'agent'),modelRuntime,model,thinkingLevel:options.thinkingLevel,
    settingsManager:SettingsManager.inMemory({compaction:{enabled:false},retry:{enabled:false,provider:{maxRetries:0}},cacheWarming:'off'}),
    resourceLoader,sessionManager:SessionManager.inMemory(cwd),tools:['read','grep','find','ls',...controlTools.map(t=>t.name)],customTools:[...readTools,...controlTools]});
  let last,interrupted;const previous=session.agent.onPayload;
  session.agent.onPayload=async(payload,requestModel)=>{
    if(record.steps.length>=options.maxRequests){interrupted='request_budget';throw Error('Parent request budget reached');}
    const request=structuredClone(await previous?.(payload,requestModel)??payload);
    if(!Array.isArray(request.messages)||!Array.isArray(request.tools))throw Error('Native Pi request capture unavailable');
    record.steps.push({request});event({type:'request',index:record.steps.length-1,request});checkpoint();return request;
  };
  const unsubscribe=session.subscribe(e=>{
    if(e.type!=='message_end')return;
    const message=structuredClone(e.message);record.messages.push(message);
    if(message.role==='assistant'){
      last=message;const step=record.steps.at(-1);if(step&&!step.response)step.response=message;
      if(message.stopReason==='length'){interrupted='length';void session.abort();}
    }
    event({type:'message_end',message});checkpoint();
  });
  const timer=setTimeout(()=>{interrupted='timeout';manager.shutdown();void session.abort();},options.timeoutMs);
  checkpoint();
  try {
    await session.prompt(task.prompt+'\n\n'+options.promptSuffix);
    if(interrupted||last?.stopReason!=='stop')throw Error(interrupted??last?.errorMessage??'Incomplete teacher response');
    if(manager.list().some(a=>!a.finishedAt))throw Error('Teacher ended while a worker was still running');
    const final=await verify();record.checks=final.checks;
    record.delegationAcceptance=delegationEligibility({agents:manager.list(),messages:record.messages,events:record.delegationEvents,checks:record.checks,teacher:options.teacherIdentity});
    if(final.passed&&record.delegationAcceptance.passed)record.status='passed';
  }catch(e){record.error=String(e.message);}
  finally {
    clearTimeout(timer);manager.shutdown();await manager.wait(undefined,10000);unsubscribe();session.dispose();
    record.delegatedWorkers=structuredClone(manager.list());record.finishedAt=new Date().toISOString();
    record.trainingEligible=false;record.semanticReview='pending';checkpoint();save(join(dir,'run.json'),record);
  }
  return {record,path:join(dir,'run.json')};
}
