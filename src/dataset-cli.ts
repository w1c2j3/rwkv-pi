#!/usr/bin/env node
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { parseArgs } from "node:util";
import { buildDataset } from "./dataset.js";
import type { RunRecord } from "./harness.js";
import { digest, sha256 } from "./provenance.js";

try {
  const { values } = parseArgs({ options: {
    runs: { type: "string" }, out: { type: "string" }, "include-smoke": { type: "boolean", default: false }, help: { type: "boolean" },
  } });
  if (values.help) console.log("Usage: npm run dataset -- --runs runs/batch --out datasets/version [--include-smoke]");
  else {
    if (!values.runs || !values.out) throw new Error("--runs and --out are required.");
    const records: RunRecord[] = [];
    const inputs: { path: string; sha256: string }[] = [];
    for (const entry of (await readdir(values.runs, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      if (!entry.isDirectory()) continue;
      const path = join(values.runs, entry.name, "run.json");
      let text: string;
      try { text = await readFile(path, "utf8"); }
      catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") continue; throw error; }
      records.push(JSON.parse(text) as RunRecord);
      inputs.push({ path, sha256: sha256(text) });
    }
    if (!records.length) throw new Error("No completed run.json files found.");
    const { samples, rejected } = buildDataset(records, values["include-smoke"]);
    // A new directory makes dataset versions immutable by default.
    await mkdir(dirname(values.out), { recursive: true });
    await mkdir(values.out);
    const counts: Record<string, number> = {};
    for (const split of ["train", "validation", "test", "smoke"]) {
      const rows = samples.filter((sample) => sample.split === split);
      counts[split] = rows.length;
      await writeFile(join(values.out, `${split}.jsonl`), rows.map((sample) => JSON.stringify(sample) + "\n").join(""), { mode: 0o600 });
    }
    const manifest = { schemaVersion: 1, createdAt: new Date().toISOString(), format: "openai-messages-with-last-assistant-loss", inputs, counts, rejected, samplesSha256: digest(samples) };
    await writeFile(join(values.out, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
    console.log(JSON.stringify({ out: values.out, counts, rejectedRuns: rejected.length }));
  }
} catch (error) {
  console.error(`rwkv-pi dataset: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
