import assert from 'node:assert/strict';
import test from 'node:test';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { execFile, spawn } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
const run = promisify(execFile);
const cli = fileURLToPath(new URL('../dist/cli.js', import.meta.url));
const flags = ['--offline', '--no-approve', '--no-extensions', '--no-skills', '--no-prompt-templates', '--no-themes', '--no-session', '--thinking', 'off', '--tools', 'read,write,bash'];

function stream(res, deltas, finish = 'stop') {
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  const send = (delta, finish_reason = null) => res.write(`data: ${JSON.stringify({ id: 'fixture', object: 'chat.completion.chunk', choices: [{ index: 0, delta, finish_reason }] })}\n\n`);
  for (const delta of deltas) send(delta);
  send({}, finish); res.end('data: [DONE]\n\n');
}
function call(res, id, name, args) {
  stream(res, [{ tool_calls: [{ index: 0, id, type: 'function', function: { name, arguments: JSON.stringify(args) } }] }], 'tool_calls');
}
async function setup(t, behavior, extra = {}) {
  const root = await mkdtemp(join(tmpdir(), 'rwkv-auto-collab-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const cwd = join(root, 'workspace'), data = join(root, 'data');
  await mkdir(cwd); await mkdir(data); await writeFile(join(cwd, 'input.txt'), 'ACTUAL_WORKSPACE\n');
  const requests = [], counts = { rwkv: 0, teacher: 0 };
  const server = createServer(async (req, res) => {
    let raw = ''; for await (const chunk of req) raw += chunk;
    if (req.url.endsWith('/tokens/count')) { res.end('{"tokens":100}'); return; }
    if (req.url.endsWith('/server/stop')) { res.end('{}'); return; }
    const body = JSON.parse(raw); const role = body.model === 'teacher-test' ? 'teacher' : 'rwkv';
    requests.push({ role, body }); counts[role]++;
    behavior({ res, body, role, count: counts[role], counts });
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const url = `http://127.0.0.1:${server.address().port}/v1`;
  const env = { ...process.env };
  for (const key of Object.keys(env)) if ((key.startsWith('RWKV_') && key !== 'RWKV_PI_SOURCE') || key.startsWith('TEACHER_')) delete env[key];
  Object.assign(env, { PI_CODING_AGENT_DIR: join(root, 'agent'), PI_OFFLINE: '1', RWKV_BACKEND: 'openai', RWKV_BASE_URL: url,
    RWKV_MODEL: 'rwkv-test', RWKV_API_KEY: 'fixture', RWKV_TOOL_FORMAT: '', RWKV_PROMPT_TEMPLATE: '', RWKV_PI_DATA_ROOT: data, TEACHER_BASE_URL: url, TEACHER_MODEL: 'teacher-test', TEACHER_API_KEY: 'fixture',
    RWKV_COLLAB_SILENCE_MS: '5000', RWKV_PLUGINS: 'strong-model', ...extra });
  return { root, cwd, data, env, requests, counts, async records() {
    const base = join(data, 'runs/collaboration/automatic');
    const dirs = await readdir(base).catch(() => []);
    return Promise.all(dirs.map(async d => ({ dir: join(base, d), status: JSON.parse(await readFile(join(base, d, 'status.json'))),
      record: await readFile(join(base, d, 'record.json'), 'utf8').then(JSON.parse).catch(() => undefined) })));
  }, async print(collab = true) {
    const p = run(process.execPath, [cli, ...flags, ...(collab ? [] : ['--no-collab']), '--print', 'Read input.txt, write result.txt containing FIXED, and verify it.'], { cwd, env, timeout: 20000, maxBuffer: 2 * 1024 * 1024 });
    p.child.stdin.end(); return p;
  } };
}
function teacherSuccess({ res, count }) {
  if (count === 1) call(res, 'teacher_read', 'read', { path: 'input.txt' });
  else if (count === 2) call(res, 'teacher_write', 'write', { path: 'result.txt', content: 'FIXED\n' });
  else if (count === 3) call(res, 'teacher_verify', 'read', { path: 'result.txt' });
  else stream(res, [{ content: 'QWEN_COMPLETED_AND_VERIFIED' }]);
}

test('real pi automatically hands failed RWKV work to Qwen, keeps cwd/history/tools, records actual diff', { timeout: 30000 }, async t => {
  const f = await setup(t, args => args.role === 'rwkv' ? call(args.res, `bad${args.count}`, 'read', { path: `missing-${args.count}.txt` }) : teacherSuccess(args));
  const { stdout, stderr } = await f.print();
  assert.match(stdout, /QWEN_COMPLETED_AND_VERIFIED/, stderr);
  assert.equal(await readFile(join(f.cwd, 'result.txt'), 'utf8'), 'FIXED\n');
  assert.equal(f.counts.rwkv, 3); assert.equal(f.counts.teacher, 4);
  const firstTeacher = f.requests.find(x => x.role === 'teacher').body;
  assert.match(JSON.stringify(firstTeacher.messages), /missing-1.txt/);
  assert.match(JSON.stringify(firstTeacher.messages), /Read input.txt/);
  assert.deepEqual(firstTeacher.tools.map(t => t.function.name).sort(), ['bash', 'read', 'write']);
  const [{ record, status }] = await f.records();
  assert.equal(record.trigger.code, 'consecutive_errors'); assert.equal(status.status, 'completed_unverified');
  assert.equal(record.acceptance.verified, false); assert.equal(record.trainingUse.exported, false);
  assert.match(record.files.find(x => x.path === 'result.txt').qwenDiff, /\+FIXED/);
});

test('successful RWKV and disabled collaboration never call Qwen', { timeout: 30000 }, async t => {
  const f = await setup(t, ({res}) => stream(res, [{ content: 'RWKV_DONE' }]));
  for (const enabled of [true, false]) assert.match((await f.print(enabled)).stdout, /RWKV_DONE/);
  assert.equal(f.counts.teacher, 0);
});

test('native provider error and streaming textual repetition trigger takeover', { timeout: 60000 }, async t => {
  for (const mode of ['provider-error', 'repetition', 'silence']) {
    const f = await setup(t, args => {
      if (args.role === 'teacher') return teacherSuccess(args);
      if (mode === 'provider-error') { args.res.writeHead(400); args.res.end('{"error":{"message":"Invalid generated XML fixture"}}'); }
      else if (mode === 'repetition') stream(args.res, [{ content: 'I must inspect the same directory again before doing anything else. '.repeat(32) }]);
      // silence: leave the response pending until the watchdog cancels it
    }, mode === 'silence' ? { RWKV_COLLAB_SILENCE_MS: '800' } : {});
    assert.match((await f.print()).stdout, /QWEN_COMPLETED_AND_VERIFIED/);
    const [{ record }] = await f.records();
    assert.equal(record.trigger.code, { 'provider-error': 'model_error', repetition: 'repeated_text', silence: 'generation_silence' }[mode]);
    assert.equal(f.counts.rwkv, 1);
  }
});

test('Qwen failure is bounded: no ping-pong and no false success record', { timeout: 30000 }, async t => {
  const f = await setup(t, ({res, role, count}) => call(res, `${role}${count}`, 'read', { path: `missing-${role}-${count}.txt` }));
  await f.print().catch(() => {}); // print mode may surface the terminal abort as a nonzero exit
  assert.equal(f.counts.rwkv, 3); assert.equal(f.counts.teacher, 3);
  const [{ status, record }] = await f.records();
  assert.equal(status.status, 'stalled_or_failed'); assert.equal(record.qwen.status, 'stalled_or_failed');
});

async function rpc(t, f) {
  const child = spawn(process.execPath, [cli, ...flags, '--mode', 'rpc'], { cwd: f.cwd, env: f.env, stdio: ['pipe', 'pipe', 'pipe'] });
  t.after(() => { child.stdin.end(); child.kill('SIGTERM'); });
  const events = []; let buffer = '', errors = '';
  child.stderr.on('data', x => { errors += x; });
  child.stdout.on('data', x => {
    buffer += x;
    for (;;) { const at = buffer.indexOf('\n'); if (at < 0) break;
      const line = buffer.slice(0, at); buffer = buffer.slice(at + 1);
      try { events.push(JSON.parse(line)); } catch {}
    }
  });
  const send = data => child.stdin.write(JSON.stringify(data) + '\n');
  const until = async predicate => {
    const start = Date.now();
    while (!predicate()) {
      if (Date.now() - start > 12000 || child.exitCode !== null) throw new Error('RPC wait failed: ' + errors + '\n' + JSON.stringify(events.slice(-5)));
      await new Promise(resolve => setTimeout(resolve, 20));
    }
  };
  send({ type: 'get_state', id: 'ready' });
  await until(() => events.some(e => e.id === 'ready'));
  return { send, until, events };
}

test('RPC new task returns to RWKV after Qwen finishes; user cancellation does not start Qwen', { timeout: 30000 }, async t => {
  let waitForCancel = false;
  const f = await setup(t, args => {
    if (args.role === 'teacher') return teacherSuccess(args);
    if (waitForCancel) return;
    if (args.count <= 3) return call(args.res, `miss${args.count}`, 'read', { path: 'missing-' + args.count });
    stream(args.res, [{ content: 'RWKV_NEXT_TASK_DONE' }]);
  });
  const p = await rpc(t, f);
  p.send({ type: 'prompt', message: 'Fix result.txt.' });
  await p.until(() => p.events.some(e => e.type === 'message_end' && JSON.stringify(e).includes('QWEN_COMPLETED_AND_VERIFIED')));
  await p.until(() => p.events.filter(e => e.type === 'agent_settled').length >= 2);
  p.send({ type: 'prompt', message: 'Now answer another question.' });
  await p.until(() => p.events.some(e => e.type === 'message_end' && JSON.stringify(e).includes('RWKV_NEXT_TASK_DONE')));
  assert.equal(f.counts.teacher, 4); assert.equal(f.counts.rwkv, 4);
  waitForCancel = true;
  p.send({ type: 'prompt', message: 'Start a task which I will cancel.' });
  await p.until(() => f.counts.rwkv === 5);
  p.send({ type: 'abort', id: 'cancel' });
  await p.until(() => p.events.some(e => e.id === 'cancel'));
  await p.until(() => p.events.filter(e => e.type === 'agent_settled').length >= 4);
  assert.equal(f.counts.teacher, 4);
  const records = await f.records();
  assert.equal(records.at(-1).status.status, 'cancelled');
});

test('native Lightning XML parsing error hands off to real pi Qwen tools', { timeout: 30000 }, async t => {
  const f = await setup(t, args => {
    if (args.role === 'teacher') return teacherSuccess(args);
    // Envelope has no identifiable tool name; this is deliberately not recoverable by XML parser.
    args.res.writeHead(200, { 'content-type': 'text/event-stream' });
    args.res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: '<tool_call>broken</tool_call><EOD>' }, finish_reason: null }] })}\n\n`);
    args.res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop', matched_stop: '<EOD>' }] })}\n\n`);
    args.res.end('data: [DONE]\n\n');
  }, { RWKV_BACKEND: 'lightning' });
  assert.match((await f.print()).stdout, /QWEN_COMPLETED_AND_VERIFIED/);
  const [{ record }] = await f.records();
  assert.equal(record.trigger.code, 'model_error');
  assert.equal(f.counts.rwkv, 1);
});

test('interactive terminal explicitly enables collaboration without a persistent collaboration footer', { timeout: 30000 }, async t => {
  try { await run('tmux', ['-V']); } catch { t.skip('tmux unavailable'); return; }
  const f = await setup(t, args => args.role === 'rwkv' ? call(args.res, `bad${args.count}`, 'read', { path: `missing-${args.count}.txt` }) : teacherSuccess(args));
  const name = `rwkv-collab-test-${process.pid}-${Date.now()}`;
  t.after(() => run('tmux', ['kill-session', '-t', name]).catch(() => {}));
  const quote = value => "'" + value.replaceAll("'", "'\\''") + "'";
  const env = Object.entries(f.env).filter(([k]) => /^(RWKV_|TEACHER_|PI_)/.test(k)).map(([k,v]) => `${k}=${v}`);
  const command = ['env', ...env, process.execPath, cli, ...flags].map(quote).join(' ');
  await run('tmux', ['new-session', '-d', '-s', name, '-x', '160', '-y', '60', '-c', f.cwd, command]);
  let pane = '';
  const until = async predicate => {
    const start = Date.now();
    while (!predicate()) {
      if (Date.now() - start > 16000) throw new Error('TUI wait failed: ' + pane);
      await new Promise(resolve => setTimeout(resolve, 100));
      pane = (await run('tmux', ['capture-pane', '-p', '-J', '-S', '-', '-t', name])).stdout;
    }
  };
  await until(() => pane.includes('rwkv-test'));
  assert.doesNotMatch(pane, /协作：|自动接管已开启/);
  const buffer = run('tmux', ['load-buffer', '-b', name, '-']);
  buffer.child.stdin.end('Read input.txt and write result.txt containing FIXED.'); await buffer;
  await run('tmux', ['paste-buffer', '-p', '-d', '-b', name, '-t', name]);
  await run('tmux', ['send-keys', '-t', name, 'Enter']);
  await until(() => pane.includes('Qwen 接管结束'));
  assert.equal(await readFile(join(f.cwd, 'result.txt'), 'utf8'), 'FIXED\n');
  assert.match(pane, /result.txt/); assert.match(pane, /QWEN_COMPLETED_AND_VERIFIED/);
  assert.doesNotMatch(pane, /协作：RWKV|协作：Qwen/);
  assert.equal(f.counts.teacher, 4); assert.equal(f.counts.rwkv, 3);
});

test('a quiet running tool is not mistaken for a silent model', { timeout: 15000 }, async t => {
  const f = await setup(t, ({ res, count }) => count === 1
    ? call(res, 'slow-tool', 'bash', { command: 'sleep 1.5; printf TOOL_FINISHED' })
    : stream(res, [{ content: 'RWKV_FINISHED_AFTER_TOOL' }]), { RWKV_COLLAB_SILENCE_MS: '500' });
  assert.match((await f.print()).stdout, /RWKV_FINISHED_AFTER_TOOL/);
  assert.equal(f.counts.teacher, 0);
});

test('default RWKV launch ignores teacher credentials and never hands off a provider failure', { timeout: 30000 }, async t => {
  const f = await setup(t, ({ res }) => {
    res.writeHead(400); res.end('{"error":{"message":"RWKV_ONLY_FAILURE_FIXTURE"}}');
  }, { RWKV_PLUGINS: '', TEACHER_KIND: 'invalid-disabled-teacher', TEACHER_API_KEY_FILE: '/missing/disabled-teacher.key',
    RWKV_SUBAGENT_CONCURRENCY: 'invalid-disabled-subagents' });
  await assert.rejects(f.print(), error => {
    assert.match(error.stderr + error.stdout, /RWKV_ONLY_FAILURE_FIXTURE/);
    return true;
  });
  assert.equal(f.counts.rwkv, 1);
  assert.equal(f.counts.teacher, 0);
  assert.deepEqual(await f.records(), []);
  assert.deepEqual(f.requests[0].body.tools.map(tool => tool.function.name).sort(), ['bash', 'read', 'write']);
});
