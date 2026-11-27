import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  computeLearningRate,
  OnlineRlCluster,
  resolveClusterUrls,
} from "../dist/online-rl-client.js";
import {
  buildToolObservationText,
  computeEpisodeReward,
  hasFinalAnswerLanguageMismatch,
  runOnlineRl,
} from "../dist/online-rl.js";

/** Minimal UTF-8 byte-level mock tokenizer (each byte -> id + 1, "<EOD>" -> 0) for bit-exact RL server tests. */
function mockTokenize(text) {
  const ids = [];
  let i = 0;
  while (i < text.length) {
    if (text.startsWith("<EOD>", i)) {
      ids.push(0);
      i += 5;
      continue;
    }
    const buf = Buffer.from(text[i], "utf8");
    for (const b of buf) ids.push(b + 1);
    i++;
  }
  return ids;
}

function mockDetokenize(ids) {
  const bytes = [];
  let out = "";
  const flush = () => {
    if (bytes.length) {
      out += Buffer.from(bytes).toString("utf8");
      bytes.length = 0;
    }
  };
  for (const id of ids) {
    if (id === 0) {
      flush();
      out += "<EOD>";
    } else {
      bytes.push(id - 1);
    }
  }
  flush();
  return out;
}

/**
 * Starts a mock `rwkv_nora_tune_server_dp` rank that implements `/health`, `/v1/tokenize`,
 * `/v1/detokenize`, `/v1/completions`, `/v1/logprobs`, `/v1/train`, and `/v1/publish`
 * with exact v1.8.15 validation rules (including `global_trained_tokens` for DP mode).
 */
async function startMockRlRank({ rank, worldSize, scriptedCompletions, ctx = 16384, maxBatch = 16 }) {
  let step = 0;
  let publishedStep = 0;
  let versionSeq = 0;
  const runId = `mock-r${rank}`;
  const refVersion = `${runId}:0`;
  let currentVersion = refVersion;
  const trainRequests = [];
  const publishRequests = [];
  const logprobRequests = [];
  const completionRequests = [];

  const server = createServer((req, res) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      const send = (code, obj) => {
        const body = JSON.stringify(obj);
        res.writeHead(code, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) });
        res.end(body);
      };
      const payload = chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : {};
      if (req.method === "GET" && req.url === "/health") {
        return send(200, {
          status: "ok",
          mode: "nora",
          dtype: "bfloat16",
          gemm_mode: "bf16",
          step,
          policy_version: currentVersion,
          published_step: publishedStep,
          reference_version: refVersion,
          reference_step: 0,
          chunk: 64,
          ctx,
          vocab_size: 512,
          dp_rank: rank,
          dp_world_size: worldSize,
          max_batch: maxBatch,
          micro_bsz: 2,
          default_max_tokens: 1024,
          default_stop: ["<EOD>", "✿", "\nUser:", "\n### User"],
          policies: [{ version: currentVersion, step: publishedStep, cache_enabled: true, cached_prefixes: 0, cache_bytes: 0 }],
        });
      }
      if (req.method === "POST" && req.url === "/v1/tokenize") {
        return send(200, { tokens: mockTokenize(payload.text) });
      }
      if (req.method === "POST" && req.url === "/v1/detokenize") {
        return send(200, { text: mockDetokenize(payload.tokens) });
      }
      if (req.method === "POST" && req.url === "/v1/completions") {
        completionRequests.push(payload);
        assert.equal(payload.policy_version, currentVersion);
        const completions = payload.prompts.map((promptIds, rowIdx) => {
          const text = scriptedCompletions({ rank, rowIdx, promptText: mockDetokenize(promptIds), callIndex: completionRequests.length - 1 });
          const tokens = mockTokenize(text);
          const logprobs = tokens.map((_, k) => -0.5 - 0.01 * (k % 5));
          const matchedStop = text.endsWith("<EOD>") ? "<EOD>" : undefined;
          return {
            tokens,
            logprobs,
            text,
            finish_reason: matchedStop ? "stop" : "length",
            truncated: !matchedStop,
            ...(matchedStop ? { matched_stop: matchedStop } : { truncation_reason: "max_tokens" }),
            cached_prompt_tokens: Math.floor(promptIds.length * 0.5),
            prefill_tokens: Math.ceil(promptIds.length * 0.5),
            max_tokens: payload.max_tokens,
          };
        });
        return send(200, {
          policy_version: currentVersion,
          step: publishedStep,
          temperature: payload.temperature,
          max_tokens: payload.max_tokens,
          completions,
          cached_prompt_tokens: completions.reduce((a, c) => a + c.cached_prompt_tokens, 0),
          prefill_tokens: completions.reduce((a, c) => a + c.prefill_tokens, 0),
          generated_tokens: completions.reduce((a, c) => a + c.tokens.length, 0),
          seconds: 0.01,
        });
      }
      if (req.method === "POST" && req.url === "/v1/logprobs") {
        logprobRequests.push(payload);
        assert.equal(payload.policy_version, refVersion);
        const logprobs = payload.sequences.map((seq) =>
          seq.slice(1).map((_, k) => -0.6 - 0.01 * (k % 5)),
        );
        return send(200, {
          policy_version: refVersion,
          step: 0,
          temperature: payload.temperature,
          logprobs,
          seconds: 0.005,
        });
      }
      if (req.method === "POST" && req.url === "/v1/train") {
        trainRequests.push(payload);
        assert.equal(payload.policy_version, currentVersion);
        assert.ok(Number.isFinite(payload.learning_rate) && payload.learning_rate > 0);
        if (worldSize > 1) {
          assert.ok(Number.isSafeInteger(payload.global_trained_tokens) && payload.global_trained_tokens > 0);
        }
        let kept = 0;
        let total = 0;
        let weighted = 0;
        for (const sample of payload.samples) {
          let pos = 0;
          for (const turn of sample.turns) {
            assert.equal(turn.tokens.length, turn.loss_mask.length);
            const adv = turn.advantages ?? sample.advantages;
            for (let k = 0; k < turn.tokens.length; k++) {
              if (turn.loss_mask[k] === 1 && pos > 0) {
                kept++;
                const a = Array.isArray(adv) ? adv[k] : adv;
                const lp = -0.5 - 0.01 * (k % 5);
                weighted += a * lp;
              }
              pos++;
            }
          }
          total += pos - 1;
        }
        const denom = worldSize > 1 ? payload.global_trained_tokens : kept;
        step += 1;
        return send(200, {
          policy_version: currentVersion,
          trained_tokens: kept,
          total_tokens: total,
          learning_rate: payload.learning_rate,
          updated: true,
          publish_required: true,
          step,
          loss: -weighted / Math.max(1, denom),
          grad_norm: 0.42,
          seconds: 0.02,
        });
      }
      if (req.method === "POST" && req.url === "/v1/publish") {
        publishRequests.push(payload);
        assert.equal(payload.expected_step, step);
        publishedStep = step;
        versionSeq += 1;
        currentVersion = `${runId}:${versionSeq}`;
        return send(200, {
          policy_version: currentVersion,
          step: publishedStep,
          ...(rank === 0 && payload.save ? { directory: `/tmp/mock-rl-step-${publishedStep}` } : {}),
          ...(rank === 0 && payload.checkpoint ? { checkpoint: `/tmp/mock-rl-step-${publishedStep}/training` } : {}),
        });
      }
      send(404, { error: `not found: ${req.url}` });
    });
  });

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  return {
    url: `http://127.0.0.1:${addr.port}`,
    trainRequests,
    publishRequests,
    logprobRequests,
    completionRequests,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

test("LR schedule, cluster URL expansion, language policy, and reward shaping", () => {
  assert.deepEqual(resolveClusterUrls(undefined, "http://127.0.0.1:8850", 3), [
    "http://127.0.0.1:8850",
    "http://127.0.0.1:8851",
    "http://127.0.0.1:8852",
  ]);
  const lrCfg = { lr: 2e-5, lrMin: 2e-6, warmupSteps: 4, totalSteps: 20, schedule: "cosine" };
  assert.ok(Math.abs(computeLearningRate(1, lrCfg) - 5e-6) < 1e-12);
  assert.ok(Math.abs(computeLearningRate(4, lrCfg) - 2e-5) < 1e-12);
  assert.ok(Math.abs(computeLearningRate(20, lrCfg) - 2e-6) < 1e-12);

  // Language policy checks ONLY final answer, never <think>!
  assert.equal(hasFinalAnswerLanguageMismatch("请修复 src/app.js 里的乘法错误并返回说明", "I fixed the multiplication bug in src/app.js and ran all tests successfully."), true);
  assert.equal(hasFinalAnswerLanguageMismatch("请修复 src/app.js 里的乘法错误并返回说明", "已修复 `src/app.js` 中的乘法逻辑，并验证测试通过。"), false);
  assert.equal(hasFinalAnswerLanguageMismatch("Fix the multiplication bug in src/app.js and explain briefly.", "Fixed the bug in `src/app.js` and verified tests pass."), false);

  const rPassClean = computeEpisodeReward({
    passed: true, truncated: false, loopGuardTriggered: false, languageMismatch: false,
    editCalls: 1, editFailures: 0, completedAssistantTurns: 3, maxTurns: 8,
  });
  assert.equal(rPassClean.reward, 1.05);

  const rFailTrunc = computeEpisodeReward({
    passed: false, truncated: true, loopGuardTriggered: false, languageMismatch: false,
    editCalls: 0, editFailures: 0, completedAssistantTurns: 2, maxTurns: 8,
  });
  assert.equal(rFailTrunc.reward, -0.1);

  const obsAfterEod = buildToolObservationText(["1: const x = 1;"], ">ok</think>\n<tool_call>{}</tool_call>\n\n<EOD>", "<EOD>");
  assert.equal(obsAfterEod, "User: <tool_response>\n1: const x = 1;\n</tool_response>\n\nAssistant: <think");
});

test("End-to-end 2-rank NCCL DP Agentic Dr.GRPO + KL + Anchored Line-Edit v2 + Dynamic Sampling", async () => {
  const workDir = mkdtempSync(join(tmpdir(), "rwkv-pi-rl-test-"));
  const tasksPath = join(workDir, "tasks.jsonl");
  const outDir = join(workDir, "out");

  // Task 1: always fails both rollouts -> zero variance -> Dynamic Sampling must skip it!
  const zeroVarTask = {
    schemaVersion: 1,
    id: "rl-zero-var-skip",
    family: "rl-smoke-zero",
    split: "train",
    source: "unit-test",
    prompt: "Update value.txt so it equals IMPOSSIBLE_VALUE.",
    files: { "value.txt": "alpha\n" },
    tools: ["read", "edit"],
    checks: [{ type: "file_equals", path: "value.txt", expected: "IMPOSSIBLE_VALUE\n" }],
  };
  // Task 2: rollout 0 passes via read -> anchored edit -> final answer; rollout 1 fails -> informative variance!
  const informativeTask = {
    task: {
      schemaVersion: 1,
      id: "rl-line-edit-pass",
      family: "rl-smoke-pass",
      split: "train",
      source: "unit-test",
      prompt: "Inspect greet.txt and summarize line 2.",
      files: { "greet.txt": "line1\nhello\nline3\n" },
      tools: ["read", "edit"],
      checks: [{ type: "file_equals", path: "greet.txt", expected: "line1\nworld\nline3\n" }],
    },
    cohort: "unit-multiturn",
    followups: [{ prompt: "Now change hello to world in greet.txt using edit." }],
  };
  writeFileSync(tasksPath, [zeroVarTask, informativeTask].map((t) => JSON.stringify(t)).join("\n") + "\n");

  const scriptedCompletions = ({ rowIdx, promptText }) => {
    if (promptText.includes("IMPOSSIBLE_VALUE")) {
      return ">Cannot produce impossible value.</think>\nSkipped.<EOD>";
    }
    const toolResponses = (promptText.match(/User: <tool_response>/g) ?? []).length;
    const hasFollowup = promptText.includes("Now change hello to world");
    if (toolResponses === 0) {
      return '>First read greet.txt to inspect line numbers.</think>\n<tool_call>{"name":"read","arguments":{"path":"greet.txt"}}</tool_call>\n\n<EOD>';
    }
    if (!hasFollowup) {
      return ">Line 2 is hello; summarize for turn 1.</think>\nLine 2 of `greet.txt` is `hello`.<EOD>";
    }
    if (toolResponses === 1) {
      if (rowIdx === 0) {
        // Passing Anchored Batch Line-Edit v2 call on follow-up turn!
        return '>Line 2 is hello; replace it with world.</think>\n<tool_call>{"name":"edit","arguments":{"path":"greet.txt","edits":[{"start_line":2,"end_line":2,"start_text":"hello","new_text":"world"}]}}</tool_call>\n\n<EOD>';
      }
      // Failing rollout writes the wrong word so reward variance > 0!
      return '>Line 2 is hello; replace with wrong.</think>\n<tool_call>{"name":"edit","arguments":{"path":"greet.txt","edits":[{"start_line":2,"end_line":2,"start_text":"hello","new_text":"wrong"}]}}</tool_call>\n\n<EOD>';
    }
    return ">Edit verified; answer concisely.</think>\nUpdated `greet.txt` line 2.<EOD>";
  };

  const rank0 = await startMockRlRank({ rank: 0, worldSize: 2, scriptedCompletions });
  const rank1 = await startMockRlRank({ rank: 1, worldSize: 2, scriptedCompletions });
  const cluster = new OnlineRlCluster([rank0.url, rank1.url], 30_000);
  try {
    const summary = await runOnlineRl(cluster, {
      tasksPath,
      outputDir: outDir,
      steps: 1,
      groupSize: 2,
      tasksPerRank: 1,
      maxTurns: 6,
      maxTokensPerTurn: 512,
      temperature: 1.0,
      algorithm: "dr_grpo",
      lr: 2e-5,
      lrMin: 5e-6,
      warmupSteps: 1,
      lrSchedule: "cosine",
      klBeta: 0.03,
      klClip: 10.0,
      maxResampleAttempts: 4,
      sandbox: "none",
      editMode: "line",
      maxToolResultChars: 4000,
      toolTimeoutMs: 15_000,
      minGenHeadroom: 128,
      identicalCallLimit: 2,
      targetFailureLimit: 3,
      saveEvery: 1,
      checkpointEvery: 1,
      evalEvery: 0,
      evalLimit: 0,
      seed: 42,
    });

    assert.equal(summary.updatesCompleted, 1);
    assert.equal(summary.finalServerStep, 1);
    assert.equal(summary.lastCheckpoint, "/tmp/mock-rl-step-1/training");

    // Both DP ranks must have received /v1/logprobs, /v1/train (with identical global_trained_tokens), and /v1/publish!
    assert.equal(rank0.logprobRequests.length, 1);
    assert.equal(rank1.logprobRequests.length, 1);
    assert.equal(rank0.trainRequests.length, 1);
    assert.equal(rank1.trainRequests.length, 1);
    assert.equal(rank0.trainRequests[0].global_trained_tokens, rank1.trainRequests[0].global_trained_tokens);
    assert.ok(rank0.trainRequests[0].global_trained_tokens > 0);
    assert.equal(rank0.publishRequests.length, 1);
    assert.equal(rank1.publishRequests.length, 1);

    // Verify per-token advantages array alignment on generated assistant turns!
    for (const sample of rank0.trainRequests[0].samples) {
      assert.ok(sample.turns.length >= 6); // prompt + 3 assistant turns + 2 tool observations
      for (const t of sample.turns) {
        if (t.loss_mask[0] === 1) {
          assert.ok(Array.isArray(t.advantages), "KL-enabled generated turns must carry per-token advantages");
          assert.equal(t.advantages.length, t.tokens.length);
        } else {
          assert.equal(t.advantages, 0);
        }
      }
    }

    const metrics = readFileSync(summary.metricsPath, "utf8").trim().split("\n").map((l) => JSON.parse(l));
    assert.equal(metrics.length, 1);
    assert.equal(metrics[0].algorithm, "dr_grpo");
    assert.equal(metrics[0].passRate, 0.5);
    assert.ok(metrics[0].meanKl > 0.05 && metrics[0].meanKl < 0.15, `expected KL ~0.1, got ${metrics[0].meanKl}`);
    assert.equal(metrics[0].checkpoint, "/tmp/mock-rl-step-1/training");
  } finally {
    await cluster.close();
    await rank0.close();
    await rank1.close();
    rmSync(workDir, { recursive: true, force: true });
  }
});
