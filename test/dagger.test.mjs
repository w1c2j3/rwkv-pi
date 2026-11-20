import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { buildDataset } from "../dist/dataset.js";
import { createRwkvProvider, readConfig } from "../dist/provider.js";
import { createTeacherProvider, readTeacherConfig } from "../dist/teacher.js";
import { runDaggerTask } from "../data_pipeline/dagger_rollout.mjs";
import { packTrajectory } from "../pipeline/trajectory-packing.mjs";
import { mockServer, toolCall } from "./helpers/mock-server.mjs";

test("DAgger mode: skips teacher when RWKV student passes, and climbs out seamlessly with train=false prefix when RWKV stumbles", async () => {
  let teacherCalls = 0;
  const studentPassServer = await mockServer((_body, idx) => {
    if (idx === 1) return { deltas: [{ reasoning_content: "Write answer." }, toolCall("write", { path: "out.txt", content: "OK\n" }, "s1")], finish: "tool_calls" };
    return { deltas: [{ reasoning_content: "Done.", content: "已完成。" }] };
  });
  const teacherServer = await mockServer((body, idx) => {
    teacherCalls++;
    const rawSys = body.messages?.[0]?.content;
    const sysMsg = Array.isArray(rawSys) ? rawSys.map(b => b.text ?? "").join("") : String(rawSys ?? "");
    assert.match(sysMsg, /You are a coding assistant/, "wire request preserves original Pi system prompt");
    assert.match(sysMsg, /内部纠错示范约束/, "wire request to teacher carries climb-out guide");
    if (idx === 1) {
      return {
        deltas: [{ reasoning_content: "刚才的 edit 没有匹配上原文，先用 read 查看 calc.py 内容。" }, toolCall("read", { path: "calc.py" }, "t1")],
        finish: "tool_calls",
      };
    }
    if (idx === 2) {
      return {
        deltas: [{ reasoning_content: "确认原文为 return a - b，现在修正为 return a + b。" }, toolCall("edit", { path: "calc.py", edits: [{ oldText: "return a - b", newText: "return a + b" }] }, "t2")],
        finish: "tool_calls",
      };
    }
    if (idx === 3) {
      return {
        deltas: [{ reasoning_content: "同时完成第二轮需求，写入 DONE.txt。", }, toolCall("write", { path: "DONE.txt", content: "VERIFIED\n" }, "t3")],
        finish: "tool_calls",
      };
    }
    return { deltas: [{ reasoning_content: "全部验证完成。", content: "已将 calc.py 修正为加法并写入 DONE.txt。" }] };
  });

  // Student stumbles: tries wrong edit oldText, then prematurely stops claiming it's fixed!
  const studentFailServer = await mockServer((_body, idx) => {
    if (idx === 1) {
      return {
        deltas: [{ reasoning_content: "直接猜内容修改 calc.py。" }, toolCall("edit", { path: "calc.py", edits: [{ oldText: "NON_EXISTENT_LINE", newText: "return a + b" }] }, "bad1")],
        finish: "tool_calls",
      };
    }
    return { deltas: [{ reasoning_content: "改好了。", content: "已经修复好了！" }] };
  });

  const outDir = await mkdtemp(join(tmpdir(), "rwkv-pi-dagger-"));
  try {
    const teacherProvider = createTeacherProvider(readTeacherConfig({
      TEACHER_KIND: "qwen",
      TEACHER_BASE_URL: teacherServer.url,
      TEACHER_MODEL: "mock-deepseek",
      TEACHER_API_KEY: "mock",
    }));

    // 1. Student passes -> zero teacher calls!
    const passStudentProvider = createRwkvProvider(readConfig({
      RWKV_BACKEND: "openai",
      RWKV_BASE_URL: studentPassServer.url,
      RWKV_MODEL: "mock-rwkv",
      RWKV_HISTORY_MODE: "epoch",
    }));
    const easyTask = {
      schemaVersion: 1,
      id: "dagger-easy-1",
      family: "dagger-test-easy",
      split: "train",
      source: "unit-test",
      prompt: "Write OK to out.txt.",
      files: { "in.txt": "hello\n" },
      tools: ["read", "write"],
      checks: [{ type: "file_equals", path: "out.txt", expected: "OK\n" }],
    };
    const rEasy = await runDaggerTask(easyTask, {
      runtime: resolve("."),
      provider: teacherProvider,
      studentProvider: passStudentProvider,
      outputDir: join(outDir, "easy"),
      sandbox: "none",
    });
    assert.equal(rEasy.kind, "student_passed");
    assert.equal(teacherCalls, 0, "teacher is never called when RWKV student passes");

    // 2. Student stumbles -> Teacher climbs out seamlessly across Turn 1 + Followup Turn 2!
    const failStudentProvider = createRwkvProvider(readConfig({
      RWKV_BACKEND: "openai",
      RWKV_BASE_URL: studentFailServer.url,
      RWKV_MODEL: "mock-rwkv",
      RWKV_HISTORY_MODE: "epoch",
    }));
    const hardTask = {
      schemaVersion: 1,
      id: "dagger-hard-1",
      family: "dagger-test-hard",
      split: "train",
      source: "unit-test",
      prompt: "请修复 calc.py 中的 add(a, b) 函数。",
      files: {
        "calc.py": "def add(a, b):\n    return a - b\n",
        "test_immutable.py": "assert True\n",
      },
      tools: ["read", "write", "edit"],
      checks: [
        { type: "file_equals", path: "calc.py", expected: "def add(a, b):\n    return a + b\n" },
        { type: "file_equals", path: "DONE.txt", expected: "VERIFIED\n" },
      ],
    };
    const rHard = await runDaggerTask(hardTask, {
      runtime: resolve("."),
      provider: teacherProvider,
      studentProvider: failStudentProvider,
      followups: ["请再写入 VERIFIED 到 DONE.txt。"],
      immutableFiles: ["test_immutable.py"],
      outputDir: join(outDir, "hard"),
      sandbox: "none",
    });
    assert.equal(rHard.kind, "dagger_recovery");
    assert.equal(rHard.record.status, "passed");
    assert.equal(rHard.record.dagger.studentPrefixSteps, 2, "student's bad steps preserved as prefix");
    assert.equal(rHard.record.dagger.teacherRecoverySteps, 4, "teacher climbed out and completed all requirements");

    // Recorded steps must NOT contain the transient wire guide!
    for (const step of rHard.record.steps) {
      const rawSys = step.request.messages?.[0]?.content;
      const sys = Array.isArray(rawSys) ? rawSys.map(b => b.text ?? "").join("") : String(rawSys ?? "");
      assert.doesNotMatch(sys, /内部纠错示范约束/, "recorded training steps stay 100% free of transient teacher guide");
    }

    // Dataset compilation + continuous segment packing:
    // RWKV's bad edit is inside train=false context, while all 4 teacher climb-out steps are train=true!
    const ds = buildDataset([rHard.record]);
    assert.equal(ds.rejected.length, 0);
    assert.equal(ds.samples.length, 4, "only the 4 teacher recovery steps are supervised targets");

    const spans = packTrajectory(ds.samples, "xml");
    assert.ok(spans.length >= 1, "packs into valid XML trajectory spans");
    const untrainedText = spans.flatMap(sp => sp.segments.filter(s => !s.train).map(s => s.text)).join("");
    const trainedText = spans.flatMap(sp => sp.segments.filter(s => s.train).map(s => s.text)).join("");
    assert.match(untrainedText, /NON_EXISTENT_LINE/, "RWKV's real mistake is preserved in train=false context");
    assert.doesNotMatch(trainedText, /NON_EXISTENT_LINE/, "RWKV's mistake is NEVER trained on (train=false)");
    assert.match(trainedText, /刚才的 edit 没有匹配上原文/, "Teacher's first-person climb-out reasoning IS trained on (train=true)");
  } finally {
    await studentPassServer.close();
    await studentFailServer.close();
    await teacherServer.close();
    await rm(outDir, { recursive: true, force: true });
  }
});
