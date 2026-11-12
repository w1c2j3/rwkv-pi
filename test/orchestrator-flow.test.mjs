// Real Pi main loop and child subprocess, simulated model endpoints. Engineering validation only.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, mkdir, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const exec = promisify(execFile);

test('native planning loop delegates to RWKV, requires diff review, applies output and records actual participants', { timeout: 60000 }, async t => {
  const root = await mkdtemp(join(tmpdir(), 'orchestrator-flow-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const ws = join(root, 'workspace'), data = join(root, 'data'); await mkdir(ws); await mkdir(data);
  const teacherRequests = []; let studentRequests = 0;
  const actions = [
    ['write', { path: 'main-bypass.txt', content: 'must be blocked' }],
    ['subagent_start', { steps: ['fix'] }],
    ['subagent_plan', { goal: '创建文件', steps: [{ name: 'fix', task: '创建 out.txt，内容为 child result。', write: true }] }],
    ['subagent_start', {}], ['subagent_wait', { ids: ['rwkv-1'], timeoutSeconds: 30 }],
    ['subagent_apply', { id: 'rwkv-1' }], // cannot apply before review
    ['subagent_diff', { id: 'rwkv-1' }], ['subagent_apply', { id: 'rwkv-1' }],
    ['project_participation', {}],
  ];
  const server = createServer(async (req, res) => {
    let raw = ''; for await (const x of req) raw += x;
    if (req.url.endsWith('/models')) { res.end(JSON.stringify({ data: [{ id: 'fixture-teacher', max_model_len: 32768 }] })); return; }
    if (req.url.endsWith('/tokens/count')) { res.end('{"tokens":100}'); return; }
    if (req.url.endsWith('/server/stop')) { res.end('{}'); return; }
    const body = JSON.parse(raw), isStudent = Array.isArray(body.contents);
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    const emit = (delta, finish_reason = null) => res.write(`data: ${JSON.stringify({ id: 'fixture', object: 'chat.completion.chunk', choices: [{ index: 0, delta, finish_reason, ...(isStudent && finish_reason ? { matched_stop: '<EOD>' } : {}) }], ...(finish_reason ? { usage: { prompt_tokens: 300, completion_tokens: 20, total_tokens: 320 } } : {}) })}\n\n`);
    if (isStudent) {
      studentRequests++;
      const answer = studentRequests === 1 ? '<tool_call>{"name":"write","arguments":{"path":"out.txt","content":"child result\\n"}}</tool_call>' : '已创建。';
      emit({ content: '</think>\n' + answer + '<EOD>' }); emit({}, 'stop');
    } else {
      teacherRequests.push(body);
      const action = actions[teacherRequests.length - 1];
      if (action) { emit({ tool_calls: [{ index: 0, id: 'call_' + teacherRequests.length, type: 'function', function: { name: action[0], arguments: JSON.stringify(action[1]) } }] }); emit({}, 'tool_calls'); }
      else { emit({ content: 'FIXTURE_COMPLETE' }); emit({}, 'stop'); }
    }
    res.end('data: [DONE]\n\n');
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  t.after(() => new Promise(r => { server.close(r); server.closeAllConnections(); }));
  const env = { ...process.env };
  for (const k of Object.keys(env)) if (k.startsWith('TEACHER_') || k.startsWith('RWKV_')) delete env[k];
  const url = `http://127.0.0.1:${server.address().port}/v1`;
  Object.assign(env, { RWKV_PI_SOURCE: resolve('.upstream/pi'), RWKV_PI_DATA_ROOT: data,
    RWKV_BACKEND: 'lightning', RWKV_BASE_URL: url, RWKV_MODEL: 'fixture-student', RWKV_API_KEY: 'fixture',
    RWKV_CONTEXT_WINDOW: '32768', RWKV_MAX_TOKENS: '4096', RWKV_SUBAGENT_SANDBOX: 'none',
    RWKV_SUBAGENT_MAX_REQUESTS: '2', RWKV_SUBAGENT_TIMEOUT_SECONDS: '30',
    TEACHER_KIND: 'qwen', TEACHER_MODEL: 'fixture-teacher', TEACHER_BASE_URL: url, TEACHER_API_KEY: 'fixture',
    TEACHER_CONTEXT_WINDOW: '32768', TEACHER_MAX_TOKENS: '4096',
    PI_CODING_AGENT_DIR: join(root, 'agent'), PI_OFFLINE: '1' });
  const cli = fileURLToPath(new URL('../dist/orchestrate-cli.js', import.meta.url));
  const p = exec(process.execPath, [cli, '--mode', 'json', '-p', '--no-session', '--offline', '--no-approve', '--no-extensions', '--no-skills', '--no-context-files',
    '--tools', [...new Set(actions.map(x => x[0]))].join(','), '帮我创建一个小文件。'], { cwd: ws, env, timeout: 45000, maxBuffer: 8 << 20 });
  p.child.stdin.end(); const { stdout, stderr } = await p;
  assert.match(stdout, /FIXTURE_COMPLETE/, stderr);
  assert.equal(await readFile(join(ws, 'out.txt'), 'utf8'), 'child result\n');
  await assert.rejects(readFile(join(ws, 'main-bypass.txt')));
  assert.equal(studentRequests, 2);
  const observation = n => teacherRequests[n].messages.filter(m => m.role === 'tool').at(-1).content;
  assert.match(observation(1), /主模型只负责/);
  assert.match(observation(2), /First call subagent_plan/);
  assert.match(observation(6), /Inspect the current subagent_diff/);
  const participation = JSON.parse(observation(9));
  assert.equal(participation.models.rwkv.toolCalls, 1);
  assert.equal(participation.models.teacher.toolCalls, 0);
  assert.equal(participation.fileEditEvents.rwkv, 1);
  assert.equal(participation.subagentStates.completed, 1);
  const projectDirs = await readdir(join(data, 'runs/participation'));
  assert.equal(projectDirs.length, 1);
});

test('opt-in subagents run a real RWKV main loop and RWKV child without teacher configuration', { timeout: 60000 }, async t => {
  const root = await mkdtemp(join(tmpdir(), 'rwkv-plugin-flow-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const ws = join(root, 'workspace'), data = join(root, 'data'); await mkdir(ws); await mkdir(data);
  const mainRequests = []; let childRequests = 0;
  const actions = [
    ['write', { path: 'parent.txt', content: 'parent can execute directly' }],
    ['subagent_plan', { goal: '创建文件', steps: [{ name: 'fix', task: '创建 out.txt，内容为 child result。', write: true }] }],
    ['subagent_start', {}], ['subagent_wait', { ids: ['rwkv-1'], timeoutSeconds: 30 }],
    ['subagent_diff', { id: 'rwkv-1' }], ['subagent_apply', { id: 'rwkv-1' }],
    ['project_participation', {}],
  ];
  const server = createServer(async (req, res) => {
    let raw = ''; for await (const x of req) raw += x;
    if (req.url.endsWith('/tokens/count')) { res.end('{"tokens":100}'); return; }
    if (req.url.endsWith('/server/stop')) { res.end('{}'); return; }
    const body = JSON.parse(raw); assert.ok(Array.isArray(body.contents), 'Only native RWKV requests are allowed');
    const isMain = body.contents[0].includes('subagent_plan');
    let text;
    if (isMain) {
      mainRequests.push(body);
      const action = actions[mainRequests.length - 1];
      text = action ? '<tool_call>' + JSON.stringify({ name: action[0], arguments: action[1] }) + '</tool_call>' : 'RWKV_ONLY_PLUGIN_COMPLETE';
    } else {
      childRequests++;
      text = childRequests === 1 ? '<tool_call>{"name":"write","arguments":{"path":"out.txt","content":"child result\\n"}}</tool_call>' : '已创建。';
    }
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    for (const choice of [{ index: 0, delta: { content: '</think>\n' + text + '<EOD>' }, finish_reason: null },
      { index: 0, delta: {}, finish_reason: 'stop', matched_stop: '<EOD>' }]) {
      res.write(`data: ${JSON.stringify({ choices: [choice], usage: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 } })}\n\n`);
    }
    res.end('data: [DONE]\n\n');
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  t.after(() => new Promise(r => { server.close(r); server.closeAllConnections(); }));
  const env = { ...process.env };
  for (const k of Object.keys(env)) if (k.startsWith('TEACHER_') || k.startsWith('RWKV_')) delete env[k];
  Object.assign(env, { RWKV_PI_SOURCE: resolve('.upstream/pi'), RWKV_PI_DATA_ROOT: data,
    RWKV_BACKEND: 'lightning', RWKV_BASE_URL: `http://127.0.0.1:${server.address().port}/v1`, RWKV_MODEL: 'fixture-rwkv', RWKV_API_KEY: 'fixture',
    RWKV_CONTEXT_WINDOW: '32768', RWKV_MAX_TOKENS: '4096', RWKV_SUBAGENT_SANDBOX: 'none',
    RWKV_SUBAGENT_MAX_REQUESTS: '2', RWKV_SUBAGENT_TIMEOUT_SECONDS: '30',
    TEACHER_KIND: 'invalid-disabled-teacher', TEACHER_API_KEY_FILE: '/missing/disabled-teacher.key',
    PI_CODING_AGENT_DIR: join(root, 'agent'), PI_OFFLINE: '1' });
  const cli = fileURLToPath(new URL('../dist/cli.js', import.meta.url));
  const p = exec(process.execPath, [cli, '--plugins', 'subagents', '--mode', 'json', '-p', '--no-session', '--offline', '--no-approve',
    '创建父任务文件，并安排一次子任务。'], { cwd: ws, env, timeout: 45000, maxBuffer: 8 << 20 });
  p.child.stdin.end(); const { stdout, stderr } = await p;
  assert.match(stdout, /RWKV_ONLY_PLUGIN_COMPLETE/, stderr);
  assert.equal(await readFile(join(ws, 'parent.txt'), 'utf8'), 'parent can execute directly');
  assert.equal(await readFile(join(ws, 'out.txt'), 'utf8'), 'child result\n');
  assert.equal(childRequests, 2); assert.equal(mainRequests.length, 8);
  assert.doesNotMatch(stdout, /"provider":"teacher"/);
  const projectDirs = await readdir(join(data, 'runs/participation'));
  const records = await readdir(join(data, 'runs/participation', projectDirs[0]));
  const record = JSON.parse(await readFile(join(data, 'runs/participation', projectDirs[0], records[0])));
  assert.equal(record.direct.teacher.responses, 0);
  assert.equal(record.direct.rwkv.responses, 8);
  assert.equal(record.subagents['rwkv-1'].state, 'completed');
});
