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
const WORKFLOW = '本次采用真实协作（两阶段闭环，9.6K滚动上下文）：第一阶段（仅当 State Ledger 中尚未记录 subagent_wait 完成时执行）最多调用 3 次只读工具（如 ls 查看目录或 read 阅读 README.md），严禁逐个通读源码，随后立即调用 subagent_plan、subagent_start、subagent_wait 将任务委派给 RWKV 子 agent 执行。第二阶段（一旦 subagent_wait 返回，或 State Ledger 已显示 subagent_wait 完成）：严禁再次调用 subagent_plan/subagent_start！若子 agent 产生有效修改可先 subagent_diff 审查再 subagent_apply；若子 agent 失败或验收未过，主工作区 read/write/edit/bash 已完全解锁，请严格按【用 read 读 1 个目标文件（<=150 行）确认 1-based 行号与首行锚点 -> 立即用 edit({"path":...,"edits":[{"start_line":...,"end_line":...,"start_text":...,"new_text":...}]}) 原子批量修改或 write 修改该文件 -> 再读下一个文件】的增量方式直接修复（严禁未读盲改或用 bash 一次性 cat 多个文件），最后调用 subagent_verify 通过独立验收并用简体中文简洁总结。';

export async function runDelegatedTask(input, options) {
  const load = name => import(pathToFileURL(join(options.runtime, 'dist', name + '.js')).href);
  const [{parseTask,taskPath},{digest,runtimeProvenance},{WorkspaceSandbox},{SubagentManager},{SubagentPlan},{foldEpochRequest},{foldWorklistRequest},{createLineModeTools,LineViewTracker},{TOLERANT_TOOL_NAMES,withTolerance}]
    = await Promise.all(['tasks','provenance','sandbox','subagents','subagent-plan','epoch','worklist','line-edit','tolerant-tools'].map(load));
  const task=parseTask(input);
  if(task.environment)throw Error('This collector currently accepts the reviewed Linux sandbox tasks only');
  const runId=randomUUID(), dir=resolve(options.outputDir,runId), cwd=join(dir,'workspace');
  mkdirSync(cwd,{recursive:true});
  for(const [name,content] of Object.entries(task.files)) {
    const p=taskPath(cwd,name);mkdirSync(dirname(p),{recursive:true});writeFileSync(p,content,{flag:'wx'});
  }
  const model=options.provider.getModels()[0];
  if(model.provider!==options.teacherIdentity?.provider||model.id!==options.teacherIdentity?.id)throw Error('Configured teacher identity mismatch');
  const followups=(options.followups??[]).map(f=>typeof f==='string'?f:f?.prompt);
  if(followups.some(f=>typeof f!=='string'||!f.trim()))throw Error('Invalid follow-up prompt');
  const effectiveEditMode=options.editMode??options.provider.rwkvPiEditMode??'line';
  const effectiveHistoryMode=options.historyMode??options.provider.rwkvPiHistoryMode??'full';
  const effectiveEpochConfig=options.epoch??options.provider.rwkvPiEpochConfig;
  const effectiveWorklistConfig=options.worklist??options.provider.rwkvPiWorklistConfig;
  const sandbox=new WorkspaceSandbox(cwd);await sandbox.probe();await sandbox.probeSearch(['read','write','edit','bash','grep','find','ls']);
  const agentCwd=sandbox.cwd;
  const {id:_id,split:_split,...definition}=task;
  const record={schemaVersion:1,runId,task,taskSha256:digest(definition),role:'teacher',
    model:{provider:model.provider,id:model.id},provenance:runtimeProvenance(),isolation:sandbox.kind,workspacePath:agentCwd,responseAdapter:'none',
    editMode:effectiveEditMode,
    ...(options.historyMode!==undefined||effectiveHistoryMode==='epoch'||effectiveHistoryMode==='worklist'?{historyMode:effectiveHistoryMode}:{}),
    startedAt:new Date().toISOString(),finishedAt:'',budget:{maxRequests:options.maxRequests,timeoutMs:options.timeoutMs},
    generation:{contextWindow:model.contextWindow,maxTokens:model.maxTokens,thinkingLevel:options.thinkingLevel},
    status:'error',checks:[],steps:[],messages:[],promptSuffix:options.promptSuffix,appendSystemPrompt:options.appendSystemPrompt,
    collectionMode:'real-teacher-rwkv-delegation-v1',supervision:{teacher:model.id,provider:model.provider,workerOutputs:'observations_only'},
    delegatedWorkers:[],delegationEvents:[],verificationRuns:[]};
  const checkpoint=()=>save(join(dir,'checkpoint.json'),record);
  const event=x=>appendFileSync(join(dir,'events.jsonl'),JSON.stringify(x)+'\n',{mode:0o600});
  const cleanImmutables=(options.immutableFiles??[]).map(n=>String(n).replace(/^(?:\.\/)?workspace\/+/,'')).filter(n=>n&&existsSync(taskPath(cwd,n)));
  const immutable=Object.fromEntries(cleanImmutables.map(name=>[name,sha(readFileSync(taskPath(cwd,name)))]));
  const totalPrompts=1+followups.length;
  let activePromptIndex=0;
  async function verify(fullAcceptance=false) {
    const isFinalTurn=fullAcceptance||activePromptIndex>=totalPrompts-1;
    const checksToRun=isFinalTurn?task.checks:task.checks.slice(0,1);
    const checks=[];
    for(const check of checksToRun) {
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
    const passed=checks.every(c=>c.passed)&&immutableChanges.length===0;
    const result={at:Date.now(),passed,checks,immutableChanges,
      ...(!isFinalTurn&&passed?{note:'当前轮次基础测试与不可变文件校验已通过；确认已完成本轮用户需求后请直接用简体中文总结回复以进入下一轮。'}:{})};
    record.verificationRuns.push(result);checkpoint();return result;
  }
  const manager=new SubagentManager(cwd,options.childLimits,join(dir,'children'),a=>{
    event({type:'worker_state',id:a.id,state:a.state,requests:a.requests,toolCalls:a.toolCalls});
  });
  record.workerRuntime={model:options.studentModel,endpoint:options.studentEndpoint,limits:options.childLimits,innerWorkspace:manager.inner};
  const plan=new SubagentPlan();
  const controlTools=delegationTools({manager,plan,verify,recordEvent:e=>{
    record.delegationEvents.push(e);event({type:'delegation',...e});checkpoint();
  }});
  const lineTracker=effectiveEditMode==='line'?new LineViewTracker():undefined;
  const baseSandboxTools=sandbox.tools(['read','write','edit','bash','grep','find','ls']);
  const rawWorkspaceTools=effectiveEditMode==='line'
    ?createLineModeTools(baseSandboxTools,agentCwd,sandbox.fs,{tolerance:true,tracker:lineTracker})
    :baseSandboxTools.map(t=>TOLERANT_TOOL_NAMES.includes(t.name)?withTolerance(t,agentCwd,sandbox.fs):t);
  let preDelegationReads=0;
  const workspaceTools=rawWorkspaceTools.map(t=>{
    return {...t,async execute(callId,args,signal,onUpdate,ctx){
      const waited=record.delegationEvents.some(e=>e.name==='subagent_wait')&&manager.list().some(a=>a.finishedAt&&a.requests>0);
      if(!waited){
        if(['write','edit','bash'].includes(t.name)){
          throw Error('请先调用 subagent_plan、subagent_start 和 subagent_wait 将任务委派给 RWKV 子 agent 执行；待子 agent 执行结束（无论成功或失败）后，write/edit/bash 才会解锁供你兜底修复。');
        }
        if(++preDelegationReads>5){
          throw Error('前期只读探查已达 5 次上限，请不要在主 agent 逐个通读源码文件！请立即调用 subagent_plan、subagent_start 和 subagent_wait 将任务委派给 RWKV 子 agent 执行；待 subagent_wait 返回后所有工具将完全解锁。');
        }
      }
      return t.execute(callId,args,signal,onUpdate,ctx);
    }};
  });
  const extraWebTools=options.webTools??[];
  const resourceLoader={getExtensions:()=>({extensions:[],errors:[],runtime:createExtensionRuntime()}),getSkills:()=>({skills:[],diagnostics:[]}),
    getPrompts:()=>({prompts:[],diagnostics:[]}),getThemes:()=>({themes:[],diagnostics:[]}),getAgentsFiles:()=>({agentsFiles:[]}),
    getSystemPrompt:()=>undefined,getSystemPromptSource:()=>undefined,getAppendSystemPrompt:()=>[options.appendSystemPrompt,WORKFLOW],
    getAppendSystemPromptSources:()=>[],extendResources:()=>{},reload:async()=>{}};
  const modelRuntime=await ModelRuntime.create({credentials:new InMemoryCredentialStore(),modelsStore:new InMemoryModelsStore(),modelsPath:null,refreshOnCreate:false,allowModelNetwork:false});
  modelRuntime.registerNativeProvider(options.provider);
  const {session}=await createAgentSession({cwd:agentCwd,agentDir:join(dir,'agent'),modelRuntime,model,thinkingLevel:options.thinkingLevel,
    settingsManager:SettingsManager.inMemory({compaction:{enabled:false},retry:{enabled:false,provider:{maxRetries:0}},cacheWarming:'off',httpIdleTimeoutMs:1_800_000}),
    resourceLoader,sessionManager:SessionManager.inMemory(agentCwd),tools:['read','write','edit','bash','grep','find','ls',...controlTools.map(t=>t.name),...extraWebTools.map(t=>t.name)],customTools:[...workspaceTools,...controlTools,...extraWebTools]});
  let last,interrupted;const previous=session.agent.onPayload;
  session.agent.onPayload=async(payload,requestModel)=>{
    if(record.steps.length>=options.maxRequests){interrupted='request_budget';throw Error('Parent request budget reached');}
    let request=await previous?.(payload,requestModel)??payload;
    if(!options.provider.rwkvPiTransportFolded){
      if(effectiveHistoryMode==='epoch')request=foldEpochRequest(request,effectiveEpochConfig).request;
      else if(effectiveHistoryMode==='worklist')request=foldWorklistRequest(request,effectiveWorklistConfig).request;
    }
    if(Array.isArray(request.messages))lineTracker?.syncWithActiveMessages(request.messages);
    request=structuredClone(request);
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
    const prompts=[task.prompt,...followups];
    const turns=[];
    for(let i=0;i<prompts.length;i++) {
      activePromptIndex=i;
      if(i>0)plan.reset();
      last=undefined;
      const suffix=options.promptSuffix?'\n\n'+options.promptSuffix:'';
      await session.prompt(prompts[i]+suffix);
      if(interrupted||last?.stopReason!=='stop')throw Error(interrupted??last?.errorMessage??'Incomplete teacher response');
      if(manager.list().some(a=>!a.finishedAt))throw Error('Teacher ended while a worker was still running');
      turns.push({index:i+1,userPrompt:prompts[i],completed:true});
      record.conversation={allTurnsCompleted:turns.length===prompts.length,turns};
      checkpoint();
    }
    const final=await verify(true);record.checks=final.checks;
    record.delegationAcceptance=delegationEligibility({agents:manager.list(),messages:record.messages,events:record.delegationEvents,checks:record.checks,teacher:options.teacherIdentity,conversation:record.conversation});
    if(final.passed&&record.delegationAcceptance.passed)record.status='passed';
  }catch(e){record.error=String(e.message);}
  finally {
    clearTimeout(timer);manager.shutdown();await manager.wait(undefined,10000);unsubscribe();session.dispose();
    record.delegatedWorkers=structuredClone(manager.list());record.finishedAt=new Date().toISOString();
    record.trainingEligible=false;record.semanticReview='pending';checkpoint();save(join(dir,'run.json'),record);
  }
  return {record,path:join(dir,'run.json')};
}
