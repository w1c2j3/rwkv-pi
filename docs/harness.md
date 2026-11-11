# Harness and datasets

The harness runs Pi itself, not a simulation of it. Each run uses Pi's own system prompt and native tools, in an
isolated workspace, and records every request and response.

## Task files

Tasks are JSON Lines (`src/tasks.ts`). `tasks/smoke.jsonl` contains synthetic examples:

```json
{"schemaVersion":1,"id":"smoke-copy-uppercase","family":"smoke-text-transform","split":"smoke",
 "source":"synthetic fixture","prompt":"Read input.txt and write its uppercase contents to output.txt.",
 "files":{"input.txt":"hello pi\n"},"tools":["read","write"],
 "checks":[{"type":"file_equals","path":"output.txt","expected":"HELLO PI\n"}]}
```

A task can provide its starting files inline, or name a pinned Docker image (`environment.kind: "docker"`). An image
is identified by its layer digest, and an optional setup patch is applied on top. `RWKV_DOCKER_PATH_MAP` translates
bind-mount paths when the collector itself runs in a container.

## Running

```bash
npm run harness -- --tasks tasks.jsonl --out runs/batch --backend rwkv \
  [--max-requests 32] [--timeout-ms 300000] [--thinking high|off] [--sandbox bubblewrap|none]
```

`--backend teacher` runs the same tasks with the configured teacher. For each task, the run directory contains these
files:

- `run.json`: task hash, provenance, generation settings, every captured request and response, final messages,
  check results, and status
- `events.jsonl`: an append-only event log
- the workspace after the run

With bubblewrap, the workspace is mounted at `/workspace`, which is also the cwd shown in the system prompt.
For RWKV runs, `run.json` records `toolTolerance: "rwkv-tolerant-tools-v1"` when tolerant tool execution was on (see
[protocol](protocol.md#recall-tolerant-tools)) and `loopGuard.targetFailureLimit`.

The harness turns off compaction and retries. The request budget and timeout are recorded as interruptions, not as
successes.

## From runs to training rows

1. `npm run dataset -- --runs runs/batch --out datasets/version` extracts the completed runs that passed their
   checks. Smoke tasks are excluded unless you pass `--include-smoke`.
2. `npm run tokenize -- --dataset datasets/version --out datasets/version-lightning` packs those runs with the shared
   renderer and the engine's native tokenizer (`npm run lightning:setup` builds the tokenizer). Context segments are
   not supervised. Complete assistant turns, including EOS, are the targets.

Passing its checks doesn't make a run training data. Questions, trajectories, and packed rows are counted
separately. Review trajectories for completion, scope, and real tool use before you train on them.
