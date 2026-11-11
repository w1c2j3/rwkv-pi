import { createHash } from "node:crypto";
import { envInteger } from "./config.js";
import { canonical, REPEAT_GUARD_PREFIX } from "./loop-guard.js";

export interface StallReason { code: string; detail: string }
export interface StallLimits {
  consecutiveErrors: number; cycleRepeats: number; maxCycle: number;
  rwkvTurns: number; teacherTurns: number; silenceMs: number; phaseMs: number;
}
export function stallLimits(env: NodeJS.ProcessEnv = process.env): StallLimits {
  const number = (key: string, fallback: number) => envInteger(env, key, fallback, 1);
  return { consecutiveErrors: number("RWKV_COLLAB_ERROR_LIMIT", 3), cycleRepeats: 3, maxCycle: 6,
    rwkvTurns: number("RWKV_COLLAB_MAX_TURNS", 48), teacherTurns: number("RWKV_COLLAB_QWEN_MAX_TURNS", 96),
    silenceMs: number("RWKV_COLLAB_SILENCE_MS", 180_000), phaseMs: number("RWKV_COLLAB_PHASE_MS", 1_800_000) };
}

const fingerprint = (value: unknown) => createHash("sha256").update(canonical(value)).digest("hex");
export function prose(message: any, thinking = false): string {
  return (message?.content ?? []).filter((b: any) => b.type === (thinking ? "thinking" : "text"))
    .map((b: any) => thinking ? b.thinking : b.text).join("");
}
const normalized = (s: string) => s.replace(/\s+/g, " ").trim();

/** Conservative periodic prose detector. Excludes fenced code and requires >= 384 repeated chars.
 * Deliberately ignores tool JSON: repeated strings inside code/data are often intentional. */
export function repeatedProse(text: string): StallReason | undefined {
  const tail = normalized(text.replace(/```[\s\S]*?(?:```|$)/g, "").replace(/~~~[\s\S]*?(?:~~~|$)/g, "")).slice(-8192);
  if (tail.length < 384) return;
  for (let width = 8; width <= Math.min(512, Math.floor(tail.length / 4)); width++) {
    const unit = tail.slice(-width);
    if (!/[\p{L}\p{N}]/u.test(unit)) continue;
    let count = 1;
    while (count * width < tail.length && tail.slice(-(count + 1) * width, -count * width) === unit) count++;
    if (count >= 4 && count * width >= 384) return { code: "repeated_text", detail: `同一段文字连续重复 ${count} 次（每段 ${width} 字符）。` };
  }
}

/** One instance per participant per user task. Results, not just commands, define a loop. */
export class StallDetector {
  private observations: string[] = [];
  private seen = new Set<string>();
  private failures = 0;
  private turns = 0;
  private noNewEvidence = 0;
  private texts: string[] = [];
  constructor(readonly limits: StallLimits = stallLimits(), readonly maxTurns = limits.rwkvTurns) {}

  turn(message: any, results: readonly any[]): StallReason | undefined {
    if (message?.role !== "assistant" || message.stopReason === "aborted") return;
    this.turns++;
    const calls = (message.content ?? []).filter((p: any) => p.type === "toolCall");
    let fresh = false;
    for (const result of results) {
      const call = calls.find((c: any) => c.id === result.toolCallId);
      if (!call) continue;
      const text = prose(result);
      if (text.startsWith(REPEAT_GUARD_PREFIX)) return { code: "identical_call", detail: text };
      this.failures = result.isError ? this.failures + 1 : 0;
      if (this.failures >= this.limits.consecutiveErrors) return { code: "consecutive_errors", detail: `连续 ${this.failures} 个工具调用失败，包含参数变化后的重试。` };
      const key = fingerprint([call.name, call.arguments, result.isError ?? false, result.content]);
      fresh ||= !this.seen.has(key);
      this.seen.add(key);
      this.observations.push(key);
      this.observations = this.observations.slice(-this.limits.maxCycle * this.limits.cycleRepeats);
      for (let width = 1; width <= this.limits.maxCycle; width++) {
        const count = width * this.limits.cycleRepeats;
        if (this.observations.length < count) continue;
        const tail = this.observations.slice(-count);
        if (tail.every((value, i) => value === tail[i % width])) return {
          code: width === 1 ? "identical_call" : "tool_cycle", detail: `${width} 步工具操作及返回结果构成的循环已重复 ${this.limits.cycleRepeats} 次。`,
        };
      }
    }
    if (message.stopReason === "error") return { code: "model_error", detail: String(message.errorMessage ?? "模型请求或工具调用解析失败").slice(0, 1200) };
    if (message.stopReason === "length") return { code: "output_limit", detail: "模型输出达到上限，当前回复不完整。" };
    const text = prose(message);
    const repeated = repeatedProse(text) ?? repeatedProse(prose(message, true));
    if (repeated) return repeated;
    this.noNewEvidence = fresh ? 0 : this.noNewEvidence + 1;
    this.texts.push(normalized(text)); this.texts = this.texts.slice(-3);
    if (this.noNewEvidence >= 3 && this.texts.length === 3 && this.texts[0]!.length >= 60 && this.texts.every(t => t === this.texts[0])) {
      return { code: "repeated_answer", detail: "连续三轮重复同一段回答，工具没有提供新观察。" };
    }
    if (!calls.length && !text.trim()) return { code: "empty_answer", detail: "模型没有调用工具，也没有给出可见回答。" };
    // A final answer after unresolved tool failures must explicitly admit inability before this
    // heuristic fires. A legitimate 'file does not exist' result alone is not a failure to finish.
    if (!calls.length && this.failures > 0 && /无法(?:继续|完成|访问|找到)|不能(?:继续|完成)|未能(?:完成|找到)|找不到|请(?:提供|确认).{0,30}(?:路径|目录)|\b(?:cannot|can't|unable to) (?:complete|continue|proceed|find|access)\b/i.test(text)) {
      return { code: "gave_up_after_error", detail: "工具错误尚未解决，模型即表示无法完成或找不到路径并结束。" };
    }
    if (calls.length && this.turns >= this.maxTurns) return { code: "turn_budget", detail: `已执行 ${this.turns} 轮，尚未给出最终答案。` };
    if (calls.length && this.noNewEvidence >= 12) return { code: "no_new_evidence", detail: "连续十二轮工具操作未产生新的调用结果。" };
  }
}
