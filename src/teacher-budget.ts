// Interactive DeepSeek shares the remaining allocation of the original CNY 10 trial.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import type { Provider } from "@earendil-works/pi-ai";
import { envInteger } from "./config.js";

export async function budgetedTeacher(original: Provider<"openai-completions">) {
  const path = process.env.TEACHER_BUDGET_LEDGER;
  const allocationPath = process.env.TEACHER_BUDGET_ALLOCATION;
  const modulePath = process.env.TEACHER_BUDGET_MODULE;
  if (!path || !allocationPath || !modulePath) throw new Error("Launch DeepSeek with scripts/run-deepseek-subagents.py (budget lock required).");
  const allocation = JSON.parse(readFileSync(allocationPath, "utf8"));
  if (!Number.isSafeInteger(allocation.priorChargedMicroCny) || allocation.priorChargedMicroCny < 0 ||
      !Number.isSafeInteger(allocation.allocatedMicroCny) || allocation.allocatedMicroCny <= 0 ||
      allocation.priorChargedMicroCny + allocation.allocatedMicroCny > 10000000) throw new Error("Invalid shared budget allocation.");
  const { PaidTrialBudget } = await import(pathToFileURL(modulePath).href);
  const budget = new PaidTrialBudget({ path, budgetCny: allocation.allocatedMicroCny / 1e6,
    arms: { "deepseek-flash": { provider: "deepseek", baseUrl: "https://api.deepseek.com", input: 2, output: 8 } },
    caps: { deepseek: 1 }, timeoutMs: 180000 });
  const fetcher = budget.forArm("deepseek-flash", process.env.TEACHER_SESSION_ID || "interactive");
  let requests = 0;
  const maxRequests = envInteger(process.env, "TEACHER_SESSION_MAX_REQUESTS", 64, 1);
  const countedFetch: typeof fetch = async (input, init) => {
    if (++requests > maxRequests) throw new Error("DeepSeek session request limit reached.");
    return fetcher(input, init);
  };
  return {
    provider: { ...original,
      stream: (m, c, o) => original.stream(m, c, Object.assign({}, o, { maxRetries: 0, fetch: countedFetch })),
      streamSimple: (m, c, o) => original.streamSimple(m, c, { ...o, maxRetries: 0, fetch: countedFetch }),
    } satisfies Provider<"openai-completions">,
    settled: () => budget.settled() as Promise<void>,
  };
}
