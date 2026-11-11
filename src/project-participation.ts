import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, realpathSync, renameSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

export type Actor = "rwkv" | "teacher";
export interface WorkCounters { responses: number; inputTokens: number; outputTokens: number; unknownUsage: number; toolCalls: number; toolErrors: number }
export const emptyCounters = (): WorkCounters => ({ responses: 0, inputTokens: 0, outputTokens: 0, unknownUsage: 0, toolCalls: 0, toolErrors: 0 });
export function countMessage(counts: WorkCounters, message: any): void {
  counts.responses++;
  const usage = message?.usage;
  if (Number.isSafeInteger(usage?.output) && usage.output > 0) {
    counts.outputTokens += usage.output;
    for (const key of ["input", "cacheRead", "cacheWrite"]) if (Number.isSafeInteger(usage[key]) && usage[key] >= 0) counts.inputTokens += usage[key];
  } else counts.unknownUsage++;
}
const sum = (a: WorkCounters, b: WorkCounters) => {
  for (const key of Object.keys(a) as (keyof WorkCounters)[]) a[key] += b[key];
};
const pct = (n: number, total: number): number | null => total ? Math.round(1000 * n / total) / 10 : null;
const display = (n: number | null) => n === null ? "暂无数据" : `${n}%`;

/** Per-process records avoid shared-file lost updates; aggregation is by canonical project directory. */
export class ProjectParticipation {
  readonly project: string;
  readonly directory: string;
  readonly file: string;
  readonly state: {
    schemaVersion: number; sessionId: string; project: string; cwd: string; mode: string; teacherModel: string;
    startedAt: string; updatedAt: string; direct: Record<Actor, WorkCounters>;
    subagents: Record<string, { counts: WorkCounters; state: string }>;
    fileEdits: { actor: Actor; path: string }[]; plans: unknown[];
  };
  constructor(cwd: string, mode: string, teacherModel: string, dataRoot = process.env.RWKV_PI_DATA_ROOT || "/mnt/f/rwkv-pi-data") {
    if (!existsSync(dataRoot)) throw new Error("Project participation requires the existing F: data root.");
    const git = spawnSync("git", ["-C", cwd, "rev-parse", "--show-toplevel"], { encoding: "utf8" });
    this.project = realpathSync(git.status === 0 ? git.stdout.trim() : cwd);
    const key = createHash("sha256").update(this.project).digest("hex").slice(0, 24);
    this.directory = join(dataRoot, "runs", "participation", key);
    mkdirSync(this.directory, { recursive: true });
    const now = new Date().toISOString(), sessionId = randomUUID();
    this.file = join(this.directory, `${sessionId}.json`);
    this.state = { schemaVersion: 1, sessionId, project: this.project, cwd: resolve(cwd), mode, teacherModel,
      startedAt: now, updatedAt: now, direct: { rwkv: emptyCounters(), teacher: emptyCounters() }, subagents: {}, fileEdits: [], plans: [] };
    this.save();
  }
  private save() {
    this.state.updatedAt = new Date().toISOString();
    writeFileSync(this.file + ".tmp", JSON.stringify(this.state, null, 2) + "\n", { mode: 0o600 });
    renameSync(this.file + ".tmp", this.file);
  }
  message(actor: Actor, message: any) { countMessage(this.state.direct[actor], message); this.save(); }
  tool(actor: Actor, error = false) { this.state.direct[actor].toolCalls++; if (error) this.state.direct[actor].toolErrors++; this.save(); }
  subagent(id: string, counts: WorkCounters, state: string) {
    const value = { counts: { ...counts }, state };
    if (JSON.stringify(this.state.subagents[id]) === JSON.stringify(value)) return;
    this.state.subagents[id] = value; this.save();
  }
  edits(actor: Actor, paths: string[]) { this.state.fileEdits.push(...paths.map(path => ({ actor, path }))); this.save(); }
  plan(plan: unknown) { this.state.plans.push(plan); this.save(); }
  summary() {
    const total = { rwkv: emptyCounters(), teacher: emptyCounters() }, fileEdits = { rwkv: 0, teacher: 0 };
    const states: Record<string, number> = {};
    let sessions = 0, unreadableRecords = 0;
    for (const name of readdirSync(this.directory).filter(n => n.endsWith(".json"))) {
      let record: typeof this.state;
      try { record = JSON.parse(readFileSync(join(this.directory, name), "utf8")); }
      catch { unreadableRecords++; continue; }
      if (record.schemaVersion !== 1 || record.project !== this.project) { unreadableRecords++; continue; }
      sessions++;
      sum(total.rwkv, record.direct.rwkv); sum(total.teacher, record.direct.teacher);
      for (const agent of Object.values(record.subagents)) { sum(total.rwkv, agent.counts); states[agent.state] = (states[agent.state] || 0) + 1; }
      for (const edit of record.fileEdits) fileEdits[edit.actor]++;
    }
    const knownShare = pct(total.rwkv.outputTokens, total.rwkv.outputTokens + total.teacher.outputTokens);
    const incomplete = total.rwkv.unknownUsage + total.teacher.unknownUsage > 0 || unreadableRecords > 0;
    return { project: this.project, sessions, models: total, subagentStates: states, fileEditEvents: fileEdits,
      rwkvOutputTokenPercent: incomplete ? null : knownShare, knownUsageOutputTokenPercent: knownShare,
      rwkvToolExecutionPercent: pct(total.rwkv.toolCalls, total.rwkv.toolCalls + total.teacher.toolCalls),
      rwkvFileEditEventPercent: pct(fileEdits.rwkv, fileEdits.rwkv + fileEdits.teacher),
      incompleteUsage: incomplete, unreadableRecords, records: this.directory,
      interpretation: "Output tokens are a generation-volume proxy across different tokenizers, not quality or useful-work share. Read-only main-agent tools count; planning/control/apply calls are excluded from execution tools. File edit events count applied file changes, including repeat edits; they are not final line ownership. Failed attempts remain counted. Historical sessions without these records are excluded." };
  }
  compact() {
    const s = this.summary();
    const generated = s.incompleteUsage ? `${display(s.knownUsageOutputTokenPercent)}（仅已知用量）` : display(s.rwkvOutputTokenPercent);
    return `RWKV 生成 ${generated} · 工具执行 ${display(s.rwkvToolExecutionPercent)} · 文件修改 ${s.fileEditEvents.rwkv}/${s.fileEditEvents.rwkv + s.fileEditEvents.teacher}`;
  }
}
