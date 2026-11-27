// On-Policy DAgger (Student RWKV pitfall -> Seamless First-Person Teacher Climb-Out).
// Runs RWKV student first; if RWKV solved the task cleanly, no teacher call is needed.
// Whenever RWKV hits a real pitfall (tool error loop, bad edit, premature stop failing checks,
// truncation, or step budget), DeepSeek continues seamlessly from the exact toolResult/user
// boundary in a copy of RWKV's actual workspace without injecting synthetic handoff prompts
// into the recorded training history. RWKV prefix steps carry supervised=false (train=false),
// while verified teacher climb-out steps carry supervised=true (train=true).
import {appendFileSync, cpSync, existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {createHash, randomUUID} from 'node:crypto';
import {dirname, join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {InMemoryCredentialStore, InMemoryModelsStore} from '@earendil-works/pi-ai';
import {createAgentSession, createExtensionRuntime, ModelRuntime, SessionManager, SettingsManager} from '@earendil-works/pi-coding-agent';

const save = (p, x) => {
  const tmp = p + '.tmp';
  writeFileSync(tmp, JSON.stringify(x, null, 2) + '\n', {mode: 0o600});
  writeFileSync(p, readFileSync(tmp), {mode: 0o600});
};
const sha = bytes => createHash('sha256').update(bytes).digest('hex');

function appendSystemGuide(msg, suffix) {
  if (!msg || (msg.role !== 'system' && msg.role !== 'developer')) return;
  if (typeof msg.content === 'string') {
    msg.content = msg.content + '\n\n' + suffix;
  } else if (Array.isArray(msg.content)) {
    msg.content = [...msg.content, {type: 'text', text: '\n\n' + suffix}];
  }
}

const TEACHER_WIRE_GUIDE =
  '【内部纠错示范约束（仅指导生成，不进入训练提示词）】：当前对话中的前序步骤来自待纠错模型的真实踩坑现场，工作区可能已存在部分改动、错误补丁、未匹配的 edit 或未通过的测试。' +
  '请你严格以【同一个助手的第一人称视角】从当前最新观察无缝接续——严禁在 <think> 思考或最终回复中出现“前任”、“上一个助手”、“学生模型”、“接管”等任何元描述，必须自然表现为你自己发现刚才的操作报错或未满足要求并立即排障修正！' +
  '当前采用 9.6K 滚动短上下文（单步结果上限 2000 tokens，仅保留最近 2 步完整输出，更早步骤折叠为 State Ledger）：请采取【定位 -> 用 read 精读 1 个目标文件（每次 <=150 行，确认当前 1-based 行号与首行内容） -> 立即用 edit({"path":...,"edits":[{"start_line":...,"end_line":...,"start_text":...,"new_text":...}]}) 原子批量修改或 write 修正 -> 运行测试验证】的增量步进策略，修改同一文件多处时尽量合并进单次 edits[] 调用，严禁未读盲改或用 bash 一次性 cat 多个文件。全部要求验证通过后用简体中文简洁总结。';

export function isStudentInfrastructureError(record) {
  const err = String(record?.error ?? '').toLowerCase();
  return ['econnrefused', 'econnreset', 'fetch failed', 'socket hang up', 'http 502', 'http 503', 'http 504', 'service unavailable']
    .some(tok => err.includes(tok));
}

export function classifyStudentFailure(record) {
  const err = String(record?.error ?? '').toLowerCase();
  if (isStudentInfrastructureError(record)) return 'infrastructure';
  if (record?.status === 'passed') return 'passed';
  if (err.includes('rwkv_tool_loop')) return 'tool_loop';
  if (err.includes('truncated tool call') || err.includes('length')) return 'truncated_generation';
  if (err.includes('xml') || err.includes('json') || err.includes('invalid arguments') || err.includes('tool is not allowed')) return 'tool_protocol';
  if (record?.status === 'budget_exhausted') return 'student_budget';
  if (record?.status === 'timeout') return 'student_timeout';
  if (record?.status === 'failed') return 'acceptance_failed';
  return record?.status || 'student_error';
}

export function prepareDaggerScene(studentRecord) {
  const messages = structuredClone(studentRecord.messages ?? []);
  const rawSteps = structuredClone(studentRecord.steps ?? []);

  // Sanitize any stray <think> tags inside student reasoning so renderPrompt never rejects prefix history.
  for (const m of messages) {
    if (m.role === 'assistant' && Array.isArray(m.content)) {
      for (const b of m.content) {
        if (b?.type === 'thinking' && typeof b.thinking === 'string') {
          b.thinking = b.thinking.replace(/<\/?think>/gi, '');
        }
      }
    }
  }

  // Drop trailing uncommitted/aborted assistant messages.
  while (messages.length > 0 && messages.at(-1)?.role === 'assistant') {
    const last = messages.at(-1);
    const hasValidStop = ['stop', 'toolUse'].includes(last.stopReason);
    const hasContent = Array.isArray(last.content) && last.content.some(b => (b.type === 'text' && b.text?.trim()) || b.type === 'toolCall');
    if (!hasValidStop || !hasContent) {
      messages.pop();
    } else {
      break;
    }
  }

  // If the student stopped prematurely (`stopReason === 'stop'`) while failing checks or before finishing turns,
  // pop that premature final assistant text response so messages end cleanly on the preceding toolResult or user prompt!
  if (messages.at(-1)?.role === 'assistant' && messages.at(-1).stopReason === 'stop') {
    messages.pop();
  }

  if (!messages.some(m => m.role === 'user') && studentRecord.task?.prompt) {
    messages.unshift({role: 'user', content: [{type: 'text', text: studentRecord.task.prompt}], timestamp: Date.now()});
  }

  // Close any committed toolCall whose toolResult was interrupted before recording.
  const observed = new Set(messages.filter(m => m.role === 'toolResult').map(m => m.toolCallId));
  const pendingIds = [];
  for (const m of messages) {
    if (m.role !== 'assistant') continue;
    for (const c of m.content ?? []) {
      if (c?.type === 'toolCall' && !observed.has(c.id)) {
        pendingIds.push(c.id);
        messages.push({
          role: 'toolResult',
          toolCallId: c.id,
          toolName: c.name,
          content: [{type: 'text', text: 'Error: Tool execution was interrupted before completion; inspect current workspace state before retrying.'}],
          isError: true,
          timestamp: Date.now(),
        });
      }
    }
  }

  // Keep only committed student steps corresponding to the retained assistant messages, marked supervised=false.
  const retainedAssistants = messages.filter(m => m.role === 'assistant').length;
  const studentSteps = rawSteps
    .filter(s => s?.response && ['stop', 'toolUse'].includes(s.response.stopReason))
    .slice(0, retainedAssistants)
    .map(s => ({
      ...s,
      supervised: false,
      actor: 'student',
      excludedFromTraining: 'student_on_policy_bad_case_prefix',
    }));

  const userCount = messages.filter(m => m.role === 'user').length;
  return {messages, studentSteps, userCount, pendingIds};
}

export async function runDaggerTask(input, options) {
  const load = name => import(pathToFileURL(join(options.runtime, 'dist', name + '.js')).href);
  const answerEvidencePath = join(options.runtime, 'pipeline', 'answer-evidence.mjs');
  const teacherHarnessPath = join(options.runtime, 'pipeline', 'teacher-harness.mjs');
  const [
    {parseTask, taskPath},
    {digest, runtimeProvenance},
    {WorkspaceSandbox},
    distHarness,
    {parseToolList},
    {hostFs, TOLERANT_TOOL_NAMES, withTolerance},
    {createLineModeTools, LineViewTracker},
    {foldEpochRequest},
    {foldWorklistRequest},
  ] = await Promise.all(['tasks', 'provenance', 'sandbox', 'harness', 'tools', 'tolerant-tools', 'line-edit', 'epoch', 'worklist'].map(load));
  const {withAnswerEvidence} = existsSync(answerEvidencePath)
    ? await import(pathToFileURL(answerEvidencePath).href)
    : {withAnswerEvidence: async (_t, _c, _r, fn) => fn()};
  const effectiveEditMode = options.editMode ?? options.provider.rwkvPiEditMode ?? 'line';
  const lineTracker = effectiveEditMode === 'line' ? new LineViewTracker() : undefined;
  const wrapWorkspaceTools = (tools, dir, fsImpl) => effectiveEditMode === 'line'
    ? createLineModeTools(tools, dir, fsImpl, {tolerance: true, tracker: lineTracker})
    : tools.map(t => (TOLERANT_TOOL_NAMES.includes(t.name) ? withTolerance(t, dir, fsImpl) : t));

  const task = parseTask(input);
  const {runTask} = task.answerEvidence && (options.sandbox ?? 'bubblewrap') !== 'none' && existsSync(teacherHarnessPath)
    ? await import(pathToFileURL(teacherHarnessPath).href)
    : distHarness;
  if (task.environment) throw Error('DAgger collector requires sandbox tasks without external Docker images');

  const followups = (options.followups ?? []).map(f => (typeof f === 'string' ? f : f?.prompt));
  if (followups.some(f => typeof f !== 'string' || !f.trim())) throw Error('Invalid follow-up prompt');
  const prompts = [task.prompt, ...followups];

  const effectiveHistoryMode = options.historyMode ?? 'epoch';
  const effectiveEpochConfig = options.epoch ?? {budgetTokens: 9600, keepRecentPairs: 2, maxResultTokens: 2000};
  const immutableList = options.immutableFiles ?? [];

  // --- Phase 1: RWKV Student Rollout ---
  const useSandbox = (options.sandbox ?? 'bubblewrap') === 'bubblewrap';
  const studentSpillDir = useSandbox ? join(resolve(options.outputDir), 'student-spill') : undefined;
  const studentWebTools = options.createWebTools ? options.createWebTools('student', {spillHostDir: studentSpillDir}) : undefined;
  let studentResult;
  if (options.acquireStudentSlot) await options.acquireStudentSlot();
  try {
    studentResult = await runTask(task, {
      provider: options.studentProvider,
      role: 'student',
      outputDir: join(options.outputDir, 'student'),
      followups,
      maxRequests: options.studentMaxRequests ?? 20,
      timeoutMs: options.studentTimeoutMs ?? 480000,
      sandbox: options.sandbox ?? 'bubblewrap',
      ...(studentSpillDir ? {spillHostDir: studentSpillDir} : {}),
      thinkingLevel: 'high',
      editMode: effectiveEditMode,
      historyMode: effectiveHistoryMode,
      epoch: effectiveEpochConfig,
      ...(studentWebTools ? {webTools: studentWebTools} : {}),
    });
  } finally {
    if (options.releaseStudentSlot) options.releaseStudentSlot();
  }

  const studentRecord = studentResult.record;
  const studentDir = dirname(studentResult.path);
  const studentWorkspace = join(studentDir, 'workspace');

  if (isStudentInfrastructureError(studentRecord)) {
    throw Error(`Student infrastructure error: ${studentRecord.error}`);
  }

  // Check if student violated immutableFiles even if checks passed
  let studentTouchedImmutable = false;
  for (const rel of immutableList) {
    if (Object.hasOwn(task.files, rel)) {
      const p = taskPath(studentWorkspace, rel);
      if (!existsSync(p) || readFileSync(p, 'utf8') !== task.files[rel]) {
        studentTouchedImmutable = true;
        break;
      }
    }
  }

  if (studentRecord.status === 'passed' && !studentTouchedImmutable) {
    return {
      kind: 'student_passed',
      record: studentRecord,
      path: studentResult.path,
      studentFailureKind: 'passed',
    };
  }

  const failureKind = studentTouchedImmutable ? 'immutable_modified' : classifyStudentFailure(studentRecord);

  // --- Phase 2: Teacher In-Place Climb-Out from RWKV's Pitfall ---
  const {messages, studentSteps, userCount, pendingIds} = prepareDaggerScene(studentRecord);
  const runId = randomUUID();
  const dir = resolve(options.outputDir, runId);
  const cwd = join(dir, 'workspace');
  mkdirSync(dir, {recursive: true});
  cpSync(studentWorkspace, cwd, {recursive: true, verbatimSymlinks: true});

  // Restore any immutable files that RWKV accidentally corrupted so acceptance remains achievable
  for (const rel of immutableList) {
    if (Object.hasOwn(task.files, rel)) {
      const p = taskPath(cwd, rel);
      mkdirSync(dirname(p), {recursive: true});
      writeFileSync(p, task.files[rel]);
    }
  }
  const immutableHashes = Object.fromEntries(
    immutableList.filter(rel => existsSync(taskPath(cwd, rel))).map(rel => [rel, sha(readFileSync(taskPath(cwd, rel)))]),
  );

  const piTools = task.tools.filter(t => t !== 'web_search' && t !== 'web_fetch');
  const teacherSpillDir = useSandbox ? join(dir, 'spill') : undefined;
  if (studentSpillDir && teacherSpillDir && existsSync(studentSpillDir)) {
    cpSync(studentSpillDir, teacherSpillDir, {recursive: true, verbatimSymlinks: true});
  }
  let sandbox;
  let agentCwd = cwd;
  if (useSandbox) {
    sandbox = new WorkspaceSandbox(cwd, undefined, undefined, teacherSpillDir);
    await sandbox.probe();
    await sandbox.probeSearch(piTools);
    agentCwd = sandbox.cwd;
  }

  const model = options.provider.getModels()[0];
  const {id: _id, split: _split, ...definition} = task;
  const record = {
    schemaVersion: 1,
    runId,
    task,
    taskSha256: digest(definition),
    role: 'teacher',
    collectionMode: 'on-policy-rwkv-dagger-v1',
    model: {provider: model.provider, id: model.id},
    provenance: runtimeProvenance(),
    isolation: sandbox?.kind ?? 'none',
    workspacePath: agentCwd,
    responseAdapter: 'none',
    editMode: effectiveEditMode,
    historyMode: effectiveHistoryMode,
    startedAt: new Date().toISOString(),
    finishedAt: '',
    budget: {maxRequests: options.maxRequests ?? 48, timeoutMs: options.timeoutMs ?? 1200000},
    generation: {contextWindow: model.contextWindow, maxTokens: model.maxTokens, thinkingLevel: options.thinkingLevel ?? 'high'},
    status: 'error',
    checks: [],
    steps: [...studentSteps],
    messages: [...messages],
    dagger: {
      studentRunId: studentRecord.runId,
      studentModel: studentRecord.model?.id,
      studentStatus: studentRecord.status,
      studentFailureKind: failureKind,
      studentError: studentRecord.error ?? null,
      studentPrefixSteps: studentSteps.length,
      teacherRecoverySteps: 0,
      interruptedToolCallsClosed: pendingIds,
    },
  };

  const checkpoint = () => save(join(dir, 'checkpoint.json'), record);
  const logEvent = e => appendFileSync(join(dir, 'events.jsonl'), JSON.stringify(e) + '\n', {mode: 0o600});
  checkpoint();

  const extraWebTools = options.createWebTools ? options.createWebTools('teacher', {spillHostDir: teacherSpillDir}) : (options.webTools ?? []);
  const settingsManager = SettingsManager.inMemory({
    compaction: {enabled: false},
    retry: {enabled: false, provider: {maxRetries: 0}},
    cacheWarming: 'off',
    httpIdleTimeoutMs: 1_800_000,
  });
  // Keep resourceLoader's system prompt 100% clean (no teacher guide in recorded steps!).
  const resourceLoader = {
    getExtensions: () => ({extensions: [], errors: [], runtime: createExtensionRuntime()}),
    getSkills: () => ({skills: [], diagnostics: []}),
    getPrompts: () => ({prompts: [], diagnostics: []}),
    getThemes: () => ({themes: [], diagnostics: []}),
    getAgentsFiles: () => ({agentsFiles: []}),
    getSystemPrompt: () => undefined,
    getSystemPromptSource: () => undefined,
    getAppendSystemPrompt: () => [],
    getAppendSystemPromptSources: () => [],
    extendResources: () => {},
    reload: async () => {},
  };

  const modelRuntime = await ModelRuntime.create({
    credentials: new InMemoryCredentialStore(),
    modelsStore: new InMemoryModelsStore(),
    modelsPath: null,
    refreshOnCreate: false,
    allowModelNetwork: false,
  });
  modelRuntime.registerNativeProvider(options.provider);

  let rawTools;
  if (sandbox) {
    rawTools = wrapWorkspaceTools(sandbox.tools(piTools), agentCwd, sandbox.fs);
  } else {
    const {
      createBashToolDefinition, createEditToolDefinition, createFindToolDefinition,
      createGrepToolDefinition, createLsToolDefinition, createReadToolDefinition, createWriteToolDefinition,
    } = await import('@earendil-works/pi-coding-agent');
    const factories = {
      read: createReadToolDefinition, write: createWriteToolDefinition, edit: createEditToolDefinition,
      bash: d => createBashToolDefinition(d, {exposeSessionEnvironment: false}),
      grep: createFindToolDefinition && createGrepToolDefinition, find: createFindToolDefinition, ls: createLsToolDefinition,
    };
    rawTools = wrapWorkspaceTools(piTools.map(n => factories[n](cwd)), cwd, hostFs);
  }
  if (lineTracker) {
    const readText = async absPath => {
      if (sandbox) return (await sandbox.fs.readFile(absPath)).toString('utf8');
      return (await hostFs.readFile(absPath)).toString('utf8');
    };
    await lineTracker.seedFromMessages(messages, agentCwd, readText);
  }
  const orderedPiTools = piTools.length ? parseToolList(piTools) : [];
  const sessionToolNames = [...orderedPiTools, ...extraWebTools.map(t => t.name)];
  const customTools = [...rawTools, ...extraWebTools];

  const sessionManager = SessionManager.inMemory(agentCwd);
  for (const m of messages) sessionManager.appendMessage(m);
  const {session} = await createAgentSession({
    cwd: agentCwd,
    agentDir: join(dir, 'agent'),
    modelRuntime,
    model,
    thinkingLevel: options.thinkingLevel ?? 'high',
    settingsManager,
    resourceLoader,
    sessionManager,
    tools: sessionToolNames,
    customTools,
  });
  session.agent.state.messages = structuredClone(messages);
  if (!messages.some(m => m.role === 'system') && session.systemPrompt) {
    session.agent.state.systemPrompt = session.systemPrompt;
  }

  let interrupted;
  let last;
  let teacherReqs = 0;
  const maxTeacherReqs = options.maxRequests ?? 48;
  const previousOnPayload = session.agent.onPayload;

  session.agent.onPayload = async (payload, reqModel) => {
    if (teacherReqs >= maxTeacherReqs) {
      interrupted = 'budget_exhausted';
      throw Error('Teacher recovery request budget exhausted.');
    }
    teacherReqs++;
    let cleanReq = (await previousOnPayload?.(payload, reqModel)) ?? payload;
    if (!options.provider.rwkvPiTransportFolded) {
      if (effectiveHistoryMode === 'epoch') cleanReq = foldEpochRequest(cleanReq, effectiveEpochConfig).request;
      else if (effectiveHistoryMode === 'worklist') cleanReq = foldWorklistRequest(cleanReq).request;
    }
    if (Array.isArray(cleanReq.messages)) {
      lineTracker?.syncWithActiveMessages(cleanReq.messages);
    }
    cleanReq = structuredClone({...cleanReq, tools: cleanReq.tools ?? []});
    record.steps.push({request: cleanReq, supervised: true, actor: 'teacher'});
    logEvent({type: 'teacher_request', index: record.steps.length - 1});
    checkpoint();

    // Inject transient teacher-only climb-out guide ONLY on the wire request sent to DeepSeek!
    const wireReq = structuredClone(cleanReq);
    const immNote = immutableList.length ? `\n注意：以下基础文件禁止修改：${immutableList.join(', ')}。` : '';
    const aeNote = task.answerEvidence
      ? '\n注意：本题验收会直接核对你的最终回复文本（精确数字、关键结论与工具调用证据），请先用工具查明事实，并在最终回复中完整、准确写出题目要求的全部答案与关键数值（严禁创建任何隐藏文件）。'
      : '';
    const guide = TEACHER_WIRE_GUIDE + immNote + aeNote + (options.appendSystemPrompt ? '\n' + options.appendSystemPrompt : '');
    if (Array.isArray(wireReq.messages)) {
      appendSystemGuide(wireReq.messages[0], guide);
    }
    return wireReq;
  };

  const unsubscribe = session.subscribe(event => {
    if (event.type !== 'message_end') return;
    const msg = structuredClone(event.message);
    record.messages.push(msg);
    if (msg.role === 'assistant') {
      last = msg;
      const step = record.steps.at(-1);
      if (step && !step.response) step.response = msg;
      if (msg.stopReason === 'length') {
        interrupted = 'error';
        record.error = 'Generation ended with length; truncated trajectories are not accepted.';
        void session.abort();
      }
    }
    checkpoint();
  });

  const timer = setTimeout(() => {
    interrupted = 'timeout';
    void session.abort();
  }, options.timeoutMs ?? 1200000);

  try {
    // 1. Seamlessly climb out of the current interrupted/failed student turn!
    last = undefined;
    await session.agent.continue();
    if (interrupted || last?.stopReason !== 'stop') {
      throw Error(interrupted ?? last?.errorMessage ?? 'Incomplete teacher climb-out');
    }

    // 2. Complete any remaining multi-turn followups!
    const startTurn = Math.max(1, userCount);
    for (let t = startTurn; t < prompts.length; t++) {
      last = undefined;
      await session.prompt(prompts[t]);
      if (interrupted || last?.stopReason !== 'stop') {
        throw Error(interrupted ?? last?.errorMessage ?? `Incomplete teacher followup turn ${t + 1}`);
      }
    }

    record.conversation = {
      allTurnsCompleted: true,
      turns: prompts.map((p, idx) => ({index: idx + 1, userPrompt: p, completed: true})),
    };

    // 3. Run independent acceptance checks (with answer-evidence support) + immutable file integrity check!
    await withAnswerEvidence(task, cwd, record, async () => {
      for (const check of task.checks) {
        try {
          if (check.type === 'file_equals') {
            const actual = readFileSync(taskPath(cwd, check.path), 'utf8');
            record.checks.push({passed: actual === check.expected, detail: 'file_equals:' + check.path});
          } else if (sandbox) {
            const r = await sandbox.exec(check.argv, {timeoutMs: check.timeoutMs});
            record.checks.push({
              passed: r.exitCode === 0 && (check.expectedStdout === undefined || r.stdout.toString() === check.expectedStdout),
              detail: (r.stdout.toString() + r.stderr.toString()).slice(0, 32768),
            });
          } else {
            const {spawnSync} = await import('node:child_process');
            const r = spawnSync(check.argv[0], check.argv.slice(1), {cwd, timeout: check.timeoutMs ?? 30000, encoding: 'utf8'});
            record.checks.push({
              passed: r.status === 0 && (check.expectedStdout === undefined || r.stdout === check.expectedStdout),
              detail: ((r.stdout ?? '') + (r.stderr ?? '')).slice(0, 32768),
            });
          }
        } catch (e) {
          record.checks.push({passed: false, detail: String(e.message ?? e)});
        }
      }
    });

    const immutableViolated = Object.entries(immutableHashes).some(([rel, h]) => {
      const p = taskPath(cwd, rel);
      return !existsSync(p) || sha(readFileSync(p)) !== h;
    });
    if (immutableViolated) {
      record.checks.push({passed: false, detail: 'immutable_files_modified'});
    }

    record.status = record.checks.length > 0 && record.checks.every(c => c.passed) ? 'passed' : 'failed';
  } catch (err) {
    record.status = interrupted ?? 'error';
    record.error = String(err?.message ?? err);
  } finally {
    clearTimeout(timer);
    unsubscribe();
    session.dispose();
    record.dagger.teacherRecoverySteps = teacherReqs;
    record.finishedAt = new Date().toISOString();
    checkpoint();
    save(join(dir, 'run.json'), record);
  }

  return {
    kind: 'dagger_recovery',
    record,
    path: join(dir, 'run.json'),
    studentPath: studentResult.path,
    studentFailureKind: failureKind,
  };
}
