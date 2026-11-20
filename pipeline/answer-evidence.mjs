// Opt-in acceptance evidence for tasks judged partly on the final reply (task.answerEvidence).
// After the agent has stopped, the actual final answer and executed tool calls are written to a
// reserved workspace path for the hidden checks only, then removed before the run record is saved,
// so later repair copies never see them. Nothing is synthesized: both come from the real messages.
import { lstat, mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";

export const EVIDENCE_DIR = ".task-evidence";

const text = (m) => (m?.content ?? []).filter((b) => b.type === "text").map((b) => b.text).join("");

/**
 * `finalAnswer` and `toolCalls` cover the whole run; `turns` splits them per user turn
 * (the last assistant text and the tool calls after each user message) for multi-turn tasks.
 */
export function answerEvidence(messages) {
    const results = new Map(messages.filter((m) => m.role === "toolResult").map((m) => [m.toolCallId, m]));
    const callsOf = (m) => (m.content ?? []).filter((b) => b.type === "toolCall").map((b) => ({
        name: b.name, arguments: b.arguments, isError: results.has(b.id) ? results.get(b.id).isError === true : null,
    }));
    const assistants = messages.filter((m) => m.role === "assistant");
    const turns = [];
    for (const m of messages) {
        if (m.role === "user") turns.push({ finalAnswer: "", toolCalls: [] });
        else if (m.role === "assistant" && turns.length) {
            turns.at(-1).finalAnswer = text(m);
            turns.at(-1).toolCalls.push(...callsOf(m));
        }
    }
    return { schemaVersion: 1, finalAnswer: text(assistants.at(-1)), toolCalls: assistants.flatMap(callsOf), turns };
}

/** Run `runChecks` with evidence present; returns its result. Failing checks are pushed to `record`. */
export async function withAnswerEvidence(task, cwd, record, runChecks) {
    if (!task.answerEvidence) return runChecks();
    const directory = join(cwd, EVIDENCE_DIR);
    // The path is reserved and never mentioned to the agent; if it exists the agent wrote it.
    if (await lstat(directory).then(() => true, () => false)) {
        record.checks.push({ passed: false, detail: `Reserved ${EVIDENCE_DIR} path existed after the run; acceptance refused.` });
        return;
    }
    await mkdir(directory);
    try {
        await writeFile(join(directory, "final.json"), JSON.stringify(answerEvidence(record.messages)) + "\n");
        return await runChecks();
    }
    finally {
        await rm(directory, { recursive: true, force: true });
    }
}

/** Resume prompts ask to finish "functionality"; that would corrupt a task judged on its reply. */
export function refuseAnswerEvidenceResume(task) {
    if (task.answerEvidence) throw new Error("Answer-evidence tasks are never resumed; start a fresh rollout.");
}
