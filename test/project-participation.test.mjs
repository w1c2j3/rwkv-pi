import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { ProjectParticipation, emptyCounters } from '../dist/project-participation.js';
import { CollaborationTrace } from '../dist/collab-trace.js';
import { SubagentPlan } from '../dist/subagent-plan.js';

test('plans dispatch independent work once and gate dependencies on full application', () => {
  const plan = new SubagentPlan();
  assert.throws(() => plan.prepare(undefined, []), /First call subagent_plan/);
  plan.set('修复并检查', [
    { name: 'fix', task: '修复', write: true },
    { name: 'inspect', task: '独立查看', write: false },
    { name: 'verify', task: '验证', write: false, dependsOn: ['fix'] },
  ], []);
  const ready = plan.prepare(undefined, []);
  assert.deepEqual(ready.map(s => s.name), ['fix', 'inspect']);
  const agents = [{ id: 'a', write: true, state: 'completed', finishedAt: 1, changes: [{ path: 'a.py' }], applied: false },
    { id: 'b', state: 'completed', finishedAt: 1, write: false }];
  plan.started(ready, agents);
  assert.throws(() => plan.prepare(['fix'], agents), /ready/);
  assert.throws(() => plan.prepare(['verify'], agents), /dependencies/);
  agents[0].applied = true;
  assert.deepEqual(plan.prepare(undefined, agents).map(s => s.name), ['verify']);
  assert.throws(() => plan.prepare(['verify', 'verify'], agents), /distinct/);
  assert.throws(() => plan.set('x', [{ name: 'a', task: 'x', write: false, dependsOn: ['a'] }], []), /earlier/);
  assert.throws(() => plan.set('x', ready, [{ state: 'running' }]), /cancel/);
});

test('failed executions can retry without reclassifying a failed write as a completed dependency', () => {
  const plan = new SubagentPlan();
  plan.set('x', [{ name: 'a', task: 'x', write: true }, { name: 'b', task: 'y', write: false, dependsOn: ['a'] }], []);
  const first = plan.prepare(undefined, []), failed = { id: 'a1', state: 'failed', finishedAt: 1, write: true, applied: true };
  plan.started(first, [failed]);
  assert.deepEqual(plan.prepare(undefined, [failed]).map(s => s.name), ['a']);
  assert.throws(() => plan.prepare(['b'], [failed]), /dependencies/);
});

test('project accounting combines concurrent sessions, counts failures, and never turns missing usage into a complete ratio', t => {
  const root = mkdtempSync(join(tmpdir(), 'participation-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const ws = join(root, 'ws'); mkdirSync(ws);
  const a = new ProjectParticipation(ws, 'plan', 'fixture', root), b = new ProjectParticipation(ws, 'repair', 'fixture', root);
  a.message('teacher', { usage: { output: 100, input: 50, cacheRead: 20 } });
  a.tool('teacher');
  const work = { ...emptyCounters(), responses: 1, outputTokens: 300, toolCalls: 3, toolErrors: 1 };
  a.subagent('r1', work, 'failed'); a.subagent('r1', work, 'failed'); // repeated status updates are not extra work
  b.message('rwkv', { usage: { output: 100, input: 10 } });
  b.tool('rwkv'); b.edits('rwkv', ['a.py']); a.edits('teacher', ['a.py']);
  const s = a.summary();
  assert.equal(s.sessions, 2); assert.equal(s.models.rwkv.outputTokens, 400);
  assert.equal(s.rwkvOutputTokenPercent, 80); assert.equal(s.rwkvToolExecutionPercent, 80);
  assert.deepEqual(s.fileEditEvents, { rwkv: 1, teacher: 1 }); assert.equal(s.subagentStates.failed, 1);
  b.message('rwkv', {});
  assert.equal(a.summary().rwkvOutputTokenPercent, null);
  assert.equal(a.summary().knownUsageOutputTokenPercent, 80);
  assert.match(a.compact(), /仅已知用量/);
  writeFileSync(join(a.directory, 'broken.json'), '{');
  assert.equal(a.summary().unreadableRecords, 1);
});

test('RWKV-only completion records changed files without fabricating a takeover', t => {
  const root = mkdtempSync(join(tmpdir(), 'participation-trace-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const ws = join(root, 'ws'); mkdirSync(ws); writeFileSync(join(ws, 'a.txt'), 'before');
  const trace = new CollaborationTrace(ws, 'fix', 'fixture', [], root);
  writeFileSync(join(ws, 'a.txt'), 'after'); writeFileSync(join(ws, '.env'), 'EXCLUDED=1');
  assert.equal(trace.finish('completed_unverified'), undefined);
  assert.deepEqual(trace.rwkvOnlyChangedFiles, ['a.txt']);
  assert.equal(trace.lastRecord, undefined);
});
