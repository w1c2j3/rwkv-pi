// Extension policy fixture, no real teacher/model calls.
import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import orchestrator from '../dist/orchestrator-extension.js';
import subagents from '../dist/subagent-extension.js';

test('main agent must plan before dispatch, cannot execute writes, and automatic continuation retains its plan', async t => {
  const root = mkdtempSync(join(tmpdir(), 'orchestrator-policy-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const prior = { ...process.env };
  t.after(() => { for (const k of Object.keys(process.env)) if (!(k in prior)) delete process.env[k]; Object.assign(process.env, prior); });
  Object.assign(process.env, { TEACHER_KIND: 'qwen', TEACHER_MODEL: 'fixture', TEACHER_API_KEY: 'fixture', TEACHER_API_KEY_FILE: '', TEACHER_BASE_URL: 'http://127.0.0.1:1/v1', RWKV_PI_DATA_ROOT: root, RWKV_SUBAGENT_SANDBOX: 'none' });
  const hooks = {}, tools = {}, ctx = { cwd: root, ui: { setStatus() {}, notify() {} }, isIdle: () => true };
  await orchestrator({ registerProvider() {}, registerCommand() {}, registerTool(tool) { tools[tool.name] = tool; },
    on(name, handler) { (hooks[name] ??= []).push(handler); } });
  const emit = async (name, e) => Promise.all((hooks[name] ?? []).map(f => f(e, ctx)));
  const invoke = (name, params) => tools[name].execute('fixture', params, undefined, undefined, ctx);
  await emit('input', { source: 'interactive', text: '修复小问题' });
  await assert.rejects(invoke('subagent_start', {}), /First call subagent_plan/);
  for (const toolName of ['bash', 'write', 'edit']) assert.equal((await emit('tool_call', { toolName }))[0].block, true);
  assert.equal((await emit('tool_call', { toolName: 'read' }))[0], undefined);
  await invoke('subagent_plan', { goal: '修复', steps: [{ name: 'fix', task: '修复', write: true }] });
  await emit('input', { source: 'extension', text: '继续' });
  // Invalid name proves the plan remains: without it this would say First call subagent_plan.
  await assert.rejects(invoke('subagent_start', { steps: ['missing'] }), /Select ready/);
  await emit('input', { source: 'interactive', text: '新任务' });
  await assert.rejects(invoke('subagent_start', {}), /First call subagent_plan/);
  await emit('message_end', { message: { role: 'assistant', usage: { output: 20, input: 10 } } });
  await emit('tool_execution_end', { toolName: 'subagent_plan', isError: false });
  await emit('tool_execution_end', { toolName: 'read', isError: false });
  const stats = (await invoke('project_participation', {})).details;
  assert.equal(stats.models.teacher.toolCalls, 1);
  assert.equal(stats.models.teacher.outputTokens, 20);
  await emit('session_shutdown', {});
});

test('RWKV subagent plugin requires no teacher, leaves native tools enabled, and attributes RWKV work correctly', async t => {
  const root = mkdtempSync(join(tmpdir(), 'rwkv-subagent-plugin-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const prior = { ...process.env };
  t.after(() => { for (const k of Object.keys(process.env)) if (!(k in prior)) delete process.env[k]; Object.assign(process.env, prior); });
  Object.assign(process.env, { TEACHER_KIND: 'invalid-disabled-teacher', TEACHER_API_KEY_FILE: '/missing/teacher.key',
    RWKV_API_KEY: 'fixture', RWKV_MODEL: 'rwkv-fixture', RWKV_PI_DATA_ROOT: root, RWKV_SUBAGENT_SANDBOX: 'none' });
  const hooks = {}, tools = {}, ctx = { cwd: root, model: { provider: 'rwkv' }, ui: { setStatus() {}, notify() {} }, isIdle: () => true };
  await subagents({ registerProvider() { assert.fail('Subagents must not register a teacher'); }, registerCommand() {},
    registerTool(tool) { tools[tool.name] = tool; }, on(name, handler) { (hooks[name] ??= []).push(handler); } });
  const emit = async (name, e) => Promise.all((hooks[name] ?? []).map(f => f(e, ctx)));
  const invoke = (name, params) => tools[name].execute('fixture', params, undefined, undefined, ctx);
  assert.deepEqual(await emit('input', { source: 'interactive', text: 'Original instruction' }), [{ action: 'continue' }]);
  for (const toolName of ['bash', 'write', 'edit']) assert.equal((await emit('tool_call', { toolName }))[0], undefined);
  await assert.rejects(invoke('subagent_start', {}), /First call subagent_plan/);
  await emit('message_end', { message: { role: 'assistant', provider: 'rwkv', usage: { output: 20, input: 10 } } });
  await emit('tool_execution_end', { toolName: 'bash', isError: false });
  const stats = (await invoke('project_participation', {})).details;
  assert.equal(stats.models.rwkv.outputTokens, 20);
  assert.equal(stats.models.rwkv.toolCalls, 1);
  assert.equal(stats.models.teacher.responses, 0);
  // When both plugins are selected, takeover output must still be attributed to the teacher.
  await emit('message_end', { message: { role: 'assistant', provider: 'teacher', usage: { output: 5, input: 10 } } });
  assert.equal((await invoke('project_participation', {})).details.models.teacher.outputTokens, 5);
  await emit('session_shutdown', {});
});
