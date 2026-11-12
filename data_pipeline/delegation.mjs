// Generic native-Pi delegation tools. Private tasks and runtime paths are supplied by the caller.
import { Type } from 'typebox';
import {existsSync, readFileSync} from 'node:fs';
import {join} from 'node:path';

const result = value => ({content: [{type: 'text', text: JSON.stringify(value, null, 2)}], details: value});
function recentToolErrors(agent) {
  if (!agent.dir || !existsSync(join(agent.dir,'events.jsonl'))) return [];
  const errors=[];
  for(const line of readFileSync(join(agent.dir,'events.jsonl'),'utf8').split('\n')) {
    let e;try{e=JSON.parse(line);}catch{continue;}
    if(e.type==='tool_execution_end'&&e.isError)errors.push({tool:e.toolName,
      message:(e.result?.content??[]).filter(x=>x.type==='text').map(x=>x.text).join('\n').slice(0,4000)});
  }
  return errors.slice(-3);
}
const summary = agent => ({id: agent.id, name: agent.name, state: agent.state, write: agent.write,
  requests: agent.requests, toolCalls: agent.toolCalls, toolErrors: agent.toolErrors,
  answer: agent.answer, error: agent.error, changes: agent.changes, applied: agent.applied === true,
  availableTools:['read','write','edit','bash','grep','find','ls'],
  workspaceAccess:agent.write?'private writable copy; review then apply':'read-only; no workspace database/report/cache creation',
  recentToolErrors:recentToolErrors(agent)});

export function delegationTools({manager, plan, verify, recordEvent}) {
  const reviewed = new Map();
  const tool = (name, description, parameters, execute) => ({name, label: name, description, parameters,
    async execute(callId, args, signal) {
      const value = await execute(args, signal);
      recordEvent({callId, name, args, result: value});
      return result(value);
    }});
  const ids = Type.Object({ids: Type.Optional(Type.Array(Type.String()))});
  return [
    tool('subagent_plan', '记录目标与中文子任务。RWKV自行调查、实现和验证；写任务用独立工作区，依赖任务须等待前置结果审查应用。',
      Type.Object({goal: Type.String(), steps: Type.Array(Type.Object({name: Type.String(), task: Type.String(),
        write: Type.Boolean({description:'任何会在项目内创建数据库、报告、缓存或修改代码的任务都设true；false仅用于静态读取。子agent可用read/write/edit/bash/grep/find/ls，没有exec。'}), dependsOn: Type.Optional(Type.Array(Type.String()))}), {minItems: 1, maxItems: 6})}),
      args => plan.set(args.goal, args.steps, manager.list())),
    tool('subagent_start', '实际启动计划中的RWKV子agent。只选就绪步骤，不能重复已完成步骤。',
      Type.Object({steps: Type.Optional(Type.Array(Type.String()))}), args => {
        const steps = plan.prepare(args.steps, manager.list());
        const delegated=steps.map(s=>({...s,task:s.task+'\n\n执行环境：工具为read、write、edit、bash、grep、find、ls。执行命令使用bash。'+(s.write?'当前为独立可写项目副本，按任务边界修改并验证。':'当前项目只读，只能调查；不要运行会生成数据库、报告或缓存的命令。')+'用简体中文简洁报告真实结果。'}));
        const started = manager.start(delegated); plan.started(steps, started);
        return started.map(summary);
      }),
    tool('subagent_status', '查看真实子agent的执行状态。', ids, args => manager.list(args.ids).map(summary)),
    tool('subagent_wait', '等待真实子agent完成，返回其答复和修改清单。等待超时不会终止子任务。',
      Type.Object({ids: Type.Optional(Type.Array(Type.String())), timeoutSeconds: Type.Optional(Type.Integer({minimum: 1, maximum: 600}))}),
      async (args, signal) => (await manager.wait(args.ids, (args.timeoutSeconds ?? 60) * 1000, signal)).map(summary)),
    tool('subagent_diff', '查看已结束子任务的实际文件差异，应用前必须审查当前差异。', Type.Object({id: Type.String()}), args => {
      if (!manager.list([args.id])[0].finishedAt) throw Error('Wait for the child before reviewing changes');
      const diff = manager.diff(args.id, 64000); if(diff.includes('… diff truncated (')){reviewed.delete(args.id);return {diff,reviewComplete:false};} reviewed.set(args.id, diff); return {diff,reviewComplete:true};
    }),
    tool('subagent_apply', '应用已审查的RWKV修改。与主工作区发生冲突的文件保留原状并报告。',
      Type.Object({id: Type.String(), files: Type.Optional(Type.Array(Type.String()))}), args => {
        if (!reviewed.has(args.id) || reviewed.get(args.id) !== manager.diff(args.id, 64000)) throw Error('Review the current child diff before applying');
        const applied = manager.apply(args.id, args.files); reviewed.delete(args.id); return applied;
      }),
    tool('subagent_verify', '在整合后的真实工作区运行独立验收。先等待子任务并应用审查通过的修改；失败后应重新定位或委派。',
      Type.Object({}), async () => {
        if (manager.list().some(a => !a.finishedAt)) throw Error('Wait for active children before verification');
        return verify();
      }),
    tool('subagent_cancel', '终止指定真实子任务，保留失败过程。', ids, args => manager.cancel(args.ids).map(summary)),
  ];
}

export function delegationEligibility({agents, messages, events, checks, teacher}) {
  const reasons = [];
  const assistants = messages.filter(m => m.role === 'assistant');
  if (!assistants.length || assistants.at(-1).stopReason !== 'stop' || assistants.some(m => !['stop','toolUse'].includes(m.stopReason))) reasons.push('incomplete_parent');
  if (!teacher || !assistants.every(m => m.provider === teacher.provider && m.model === teacher.id)) reasons.push('mixed_parent_teacher');
  if (!checks.length || checks.some(c => !c.passed)) reasons.push('independent_acceptance_failed');
  if (!agents.some(a => a.state === 'completed' && a.requests > 0 && a.toolCalls >= 2)) reasons.push('no_completed_real_worker');
  if (agents.some(a => !a.finishedAt)) reasons.push('unfinished_worker');
  const names = new Set(events.map(e => e.name));
  for (const name of ['subagent_plan','subagent_start','subagent_wait','subagent_verify']) if (!names.has(name)) reasons.push('missing_' + name);
  if (agents.some(a => a.write && a.applied === true) && (!names.has('subagent_diff') || !names.has('subagent_apply'))) reasons.push('missing_review_apply');
  return {passed: reasons.length === 0, reasons, teacherTargetsOnly: true,
    studentOutputsAreObservations: true, semanticReviewRequired: true};
}
