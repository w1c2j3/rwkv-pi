import type { Subagent, SubagentSpec } from "./subagents.js";

export interface PlannedStep extends SubagentSpec { name: string; write: boolean; dependsOn?: string[] }
/** A plan is concrete dispatch data, not an extra user approval step. */
export class SubagentPlan {
  private revision = 0;
  current?: { id: string; goal: string; steps: PlannedStep[]; executions: Record<string, string> };
  set(goal: string, steps: PlannedStep[], agents: Subagent[]) {
    if (agents.some(a => !a.finishedAt)) throw new Error("Wait for or cancel running subagents before changing the plan.");
    if (!goal.trim() || !steps.length || steps.length > 16) throw new Error("Give a concrete goal and 1–16 steps.");
    const seen = new Set<string>();
    for (const s of steps) {
      if (!s.name.trim() || !s.task.trim() || seen.has(s.name)) throw new Error("Each planned step needs a distinct name and complete task.");
      if (s.dependsOn?.some(d => !seen.has(d))) throw new Error("Dependencies must name earlier steps in this plan.");
      seen.add(s.name);
    }
    this.current = { id: `plan-${++this.revision}`, goal, steps: structuredClone(steps), executions: {} };
    return this.current;
  }
  reset() { this.current = undefined; }
  prepare(names: string[] | undefined, agents: Subagent[]): PlannedStep[] {
    const p = this.current;
    if (!p) throw new Error("First call subagent_plan with concrete RWKV execution steps.");
    const previous = (name: string) => agents.find(a => a.id === p.executions[name]);
    const finished = (name: string) => {
      const a = previous(name);
      return a?.state === "completed" && (!a.write || !a.changes?.length || a.applied === true);
    };
    const ready = p.steps.filter(s => {
      const a = previous(s.name);
      // Completed writes await review/application; do not rerun them as "ready" work.
      return (!a || (a.finishedAt && a.state !== "completed")) && (s.dependsOn ?? []).every(finished);
    });
    const selected = names ? names.map(n => ready.find(s => s.name === n)) : ready;
    if (!selected.length || selected.some(s => !s) || new Set(names).size !== (names?.length ?? 0)) throw new Error("Select ready, distinct steps; dependencies must complete and their changes must be applied first.");
    return selected as PlannedStep[];
  }
  started(steps: PlannedStep[], agents: Subagent[]) {
    steps.forEach((s, i) => { this.current!.executions[s.name] = agents[i]!.id; });
  }
}
