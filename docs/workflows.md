# Workflows: teacher takeover and subagents

Plain `rwkv-pi` runs RWKV alone. Two optional workflows add a stronger "teacher" model. Neither one turns on
because credentials exist: a plugin or command has to enable it explicitly.

```bash
npm run start:source -- --offline --plugins none
npm run start:source -- --offline --plugins strong-model
npm run start:source -- --offline --plugins subagents
npm run start:source -- --offline --plugins strong-model,subagents
```

`RWKV_PLUGINS` sets the default. Command-line switches apply in order, and `--collab`/`--no-collab` and
`--subagents`/`--no-subagents` are aliases.

## Teacher configuration

| Variable | Purpose |
| --- | --- |
| `TEACHER_KIND` | `qwen` (default; any OpenAI-compatible endpoint) or `deepseek` |
| `TEACHER_BASE_URL`, `TEACHER_MODEL` | Required teacher endpoint and model |
| `TEACHER_API_KEY` / `TEACHER_API_KEY_FILE` | Teacher credential |
| `TEACHER_CONTEXT_WINDOW`, `TEACHER_MAX_TOKENS` | Teacher budgets |
| `TEACHER_REASONING_EFFORT` | `high` (default) or `low` for `deepseek` |

The `deepseek` profile accepts only the official `deepseek-flash` endpoint. It runs through a spending ledger
(`TEACHER_BUDGET_LEDGER`, `TEACHER_BUDGET_ALLOCATION`, `TEACHER_BUDGET_MODULE`; see `src/teacher-budget.ts`). The
launcher that supplies those values is not part of this repository. `TEACHER_SESSION_MAX_REQUESTS` (default 64) caps
the requests in one session.

`npm run step` starts a separate StepFun provider, configured with `STEP_API_KEY`/`STEP_API_KEY_FILE`, `STEP_MODEL`,
`STEP_BILLING` (`api` or `plan`), `STEP_REASONING_EFFORT`, `STEP_CONTEXT_WINDOW`, and `STEP_MAX_TOKENS`.

## Strong-model takeover (`strong-model`)

RWKV starts every user task. The stall detector (`src/stall-detector.ts`) hands the session to the teacher when
RWKV hits any of these:

- repeated tool errors
- repeated call cycles
- periodic repeated text
- abandonment after failures
- a run of turns that produce no new evidence
- one of the turn or time limits below

| Variable | Default |
| --- | --- |
| `RWKV_COLLAB_ERROR_LIMIT` | 3 consecutive errors |
| `RWKV_COLLAB_MAX_TURNS` | 48 RWKV turns |
| `RWKV_COLLAB_QWEN_MAX_TURNS` | 96 teacher turns |
| `RWKV_COLLAB_SILENCE_MS` | 180000 |
| `RWKV_COLLAB_PHASE_MS` | 1800000 |

The teacher continues in the same Pi session and workspace with the same tool permissions, and the next user task
goes back to RWKV. Cancelling a task never launches the teacher. `/collab off` disables takeover for the current
session.

A stall signal is a heuristic, not proof that the answer is wrong. Each takeover writes a trace under
`$RWKV_PI_DATA_ROOT/runs/collaboration/` with these parts:

- workspace snapshots before the task, at handoff, and at the end
- an event log
- a report of which RWKV lines the teacher kept or replaced

These traces are diagnostics, marked `trainingEligible: false`.

`npm run collab` runs the same comparison offline on task files. Its `report` and `scan` subcommands summarize
existing runs.

## Subagents

**`subagents` plugin.** RWKV is both the main agent and the child. The main agent keeps its native tools and can
delegate work to RWKV children. No teacher is needed.

**`npm run orchestrate`.** The teacher is the main agent. It plans, reviews, and applies changes, but cannot execute
commands or write files itself. RWKV children do the execution.

- `subagent_plan` records a goal and 1–16 named steps, each with a write flag and optional dependencies on earlier
  steps.
- `subagent_start` dispatches the steps that are ready. Independent steps may run concurrently.
- A step whose dependencies are not finished and applied is not ready. A completed write that is awaiting review is
  never rerun.
- Write results must be reviewed with `subagent_diff` before `subagent_apply`. A changed proposal needs a new review,
  and conflicts become new merge work rather than being silently overwritten.

Children run in isolated copies of the workspace. By default this is bubblewrap with the workspace mounted at
`/srv/workspace`, and teacher credentials are not passed to children.

| Variable | Default |
| --- | --- |
| `RWKV_SUBAGENT_SANDBOX` | `bwrap` (`none` only for local tests) |
| `RWKV_SUBAGENT_MOUNT` | `/srv/workspace` (must be under `/srv` or `/tmp`) |
| `RWKV_SUBAGENT_CONCURRENCY` | 16 |
| `RWKV_SUBAGENT_MAX_TASKS` | 64 |
| `RWKV_SUBAGENT_MAX_REQUESTS` | 32 per child |
| `RWKV_SUBAGENT_TIMEOUT_SECONDS` | 1200 |
| `RWKV_ORCH_LENGTH_CONTINUES` | 3 continuations after a length stop |

## Participation reporting

`/participation` and the `project_participation` tool report RWKV's share of three kinds of activity, for each project:

- generated output tokens
- native tool executions
- applied file-edit events

Records are written under `$RWKV_PI_DATA_ROOT/runs/participation/`. The numbers measure activity, not correctness or
final code ownership. Tokenizer differences, missing usage data, and retries limit their precision.
