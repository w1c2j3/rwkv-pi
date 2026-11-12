# Two output modes

Question authoring and answering are separate. This interface reads complete bundles from private authors; it does
not generate question contents. DeepSeek remains the default production teacher. Explicit teacher profiles may solve
the same frozen questions independently. This page documents current behavior.

## 1. Questions only

```bash
python3 -B -m data_pipeline produce --mode questions \
  --source /mnt/f/rwkv-pi-data/datasets/PRIVATE/bundles \
  --out /mnt/f/rwkv-pi-data/datasets/EXPORT/questions
```

The output contains `manifest.json` and `bundles/*.json`, preserving the Chinese task, environment, checks, and future
followups. It freezes original bytes without running checks, solving, or calling an API. Structural completeness is
not semantic approval; the manifest retains review status. Same-ID versions require explicit source selection and do
not become extra questions.

## 2. Questions and trajectory training data

Plan collection from one frozen package with one or more teachers:

```bash
python3 -B -m data_pipeline produce --mode trajectories \
  --questions /mnt/f/rwkv-pi-data/datasets/EXPORT/questions \
  --profile /mnt/f/rwkv-pi-data/private/teacher-a.json \
  --profile /mnt/f/rwkv-pi-data/private/teacher-b.json \
  --out /mnt/f/rwkv-pi-data/datasets/EXPORT/attempt-01
```

This writes `plan.json` by default. Only `--execute` invokes the configured native Pi collector. Each teacher receives
an independent initial workspace. Within a question, followups reuse the same session and workspace and are sent only
when their turn arrives. The question hash includes the full task and followups. The profile hash includes model,
endpoint, reasoning settings, prompt, and runtime identity, excluding secrets. `profile.example.json` is a schema
example; actual private paths, supported models, and verified prices are required for execution.

The collector currently supports **OpenAI Chat Completions-compatible APIs** through the existing native Pi runtime,
retaining its full system prompt, Chinese constraints, tool observations, and teacher output. Responses-only or
Anthropic-only models need corresponding adapters. The current native renderer/tokenizer export is XML+JSON.

### Accounting and delegation

Execution reuses the selected account's ledger and owner lock, including historical costs and unknown reservations.
Models sharing an account need entries in that ledger's price table. Changing models does not increase the account
cap. A busy production lock stops the new collector; interrupted requests are not automatically replayed.
The default DeepSeek answer allocation uses the existing CNY480 ledger, separate from author spending. Other accounts
cannot borrow it. A module's generic CNY10 trial default is not a new user authorization; an explicitly authorized
budget configuration is required for actual spending.

Real delegation uses `delegated_rollout.mjs`: a strong teacher plans/reviews and actual RWKV children execute. Child
outputs are teacher observations, not teacher-authored answers. This adapter currently supports ordinary Linux sandbox
tasks with one user request. Delegated followups are explicitly rejected. Ordinary nondelegated collection supports
sequential native multi-user-turn followups.

### Review and export

Raw results go to `teachers/<profileId>/runs`, `traces`, `raw`, and `experiment.json`. They are pending review, not
training data merely because tools ran. After independent trajectory review and native packing, list accepted entries:

```json
[
  {
    "profile": "/mnt/f/rwkv-pi-data/private/teacher-a.json",
    "run": "/mnt/f/rwkv-pi-data/datasets/EXPORT/attempt-01/teachers/PROFILE/runs/RUN/run.json",
    "review": "/mnt/f/rwkv-pi-data/datasets/EXPORT/reviews/RUN.json",
    "packedDataset": "/mnt/f/rwkv-pi-data/datasets/EXPORT/native-packed/RUN"
  }
]
```

The current exporter expects the review to bind `runSha256`, `questionSha256`, and `profileId`, identify a `reviewer`, provide specific `notes`,
set `verdict: "pass"`, record `effectiveToolCalls >= 20`, and confirm `semanticPassed`, `wirePassed`, `scopePassed`,
and `noPadding`. Judge effective calls from the actual trajectory; raw totals are not a substitute. Failed, pending,
or padded trajectories remain in raw storage.

The current exporter expects the native packer's manifest to contain original run hashes, row indexes, vocabulary/engine identity, and
`segment-end-target-v1`. This interface selects the native packed rows byte-for-byte; it does not rewrite answers.
Old runs lacking profile binding are not migrated implicitly by the current exporter; a migration/conversion path can
be added when needed.

```bash
python3 -B -m data_pipeline produce --mode trajectories \
  --questions /mnt/f/rwkv-pi-data/datasets/EXPORT/questions \
  --entries /mnt/f/rwkv-pi-data/datasets/EXPORT/reviewed-entries.json \
  --out /mnt/f/rwkv-pi-data/datasets/EXPORT/training
```

```text
training/
  manifest.json             # separate question, trajectory, and row counts
  questions/                # frozen question package
  teachers/<profileId>/
    manifest.json           # profile, source hashes, indexes, training contract
    train.jsonl             # native XML+JSON continuous training rows
    runs/                   # original teacher runs
    reviews/                # independent review evidence
```

Two teachers answering one question count as one question and two trajectories. Keep separate teacher training files
for RWKV fine-tuning comparisons. A task/environment/followup change creates a new question version; a model-setting
or Chinese-constraint change creates a new teacher profile in the current implementation.

## Comparison and engineering tests

`compare --attempts FILE --out FILE` accepts a combined list of results from teacher `experiment.json` files.
It aligns `questionId + questionSha256`, retaining teacher identity, raw/effective calls, decision rounds, user turns,
duration, and review state. Different question versions are not treated as matched comparisons.

```bash
python3 -B -m unittest data_pipeline.test_pipeline data_pipeline.test_products -v
RWKV_PI_SOURCE=/PRIVATE/PINNED_PI node scripts/run-pi-source.mjs --test data_pipeline/test_delegation.mjs
```

Optional `data_pipeline.test_collection.CollectionTest` uses a local fake SSE service with real Pi/sandbox integration.
Set `RWKV_PIPELINE_TEST_RUNTIME` and `RWKV_PI_SOURCE` first. These tests do not use real credentials or paid models,
and their fixtures are not corpus data.
