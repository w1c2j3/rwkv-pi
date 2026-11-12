# Data pipeline

The current interface supports **frozen questions** and **questions plus teacher trajectories/training data**.
The same question may be solved independently by multiple configured teachers, with separate profile identities and
training files. See [output modes](outputs.md) for commands and adapter limits.

This module is developed on the literal `data` branch in `/home/chase/GitHub/rwkv-pi-data`.
The original checkout at `/home/chase/GitHub/rwkv-pi` contains separate local runtime work; its uncommitted changes
are not all present in this worktree. Human guides are English; model-facing data stays Chinese.

## Two admission paths

The newer `produce --mode questions` entry freezes final bundles without solving them or executing reference controls.
It preserves the review state; structural completeness is not semantic approval. The subsequent trajectory product
requires actual execution, independent review, and native packing.

The original inventory/review/release commands below implement a stricter question-admission path:

```text
read-only source bundles -> versioned inventory
                                  |
                 semantic review + real controls calibration
                                  |
                     hash-bound frozen release package
                                  |
                      existing solver admission
```

This lane does not itself call paid models, control services, or train. It does not count questions as trajectories.
Its existing gates differ from those of the newer question-only product. The sections below describe current behavior.

## Inventory and versioning

- The inventory preserves original bundle bytes, SHA-256, author, source, and ID. Same-ID revisions are versions, not extra questions.
- Version selection is explicit when several coexist, rather than based on modification time. The current release
  checks bind reviews and calibration to the selected hash.
- Check exact normalized mechanisms, scenarios, prompts, and initial files. Case/width/whitespace normalization helps
  catch duplicates but cannot certify semantic novelty.
- Held-out exclusion reads family identifiers from manifests without importing held-out questions or answers.
- An inventory entry is not approval. Author self-reviews and old embedded pass claims are not independent signoff.
  New inventory state does not rewrite the original project's accepted releases or trajectories.
- Flag nonliteral deliverable paths for semantic review; literal string matching is not a substitute for reading a task.
- Validate actual baseline/reference/all-wrong-control receipts. A timeout or signal is not proof that a wrong solution
  was rejected by the checker.
- Preserve calibration receipts and hashes in `calibration-results/`. Releases retain `newBundles`, `allowedAuthors`,
  and `bundles/*.json` compatibility. `authorizedToStart=false` records pending admission; the current admission bridge checks author support and quotas.
- Release reports record the actual data-management/Agent versus other-domain mix. Current releases target 1:2;
  the later user-requested mix is 1:1.

## Commands

These commands use Python's standard library and the repository's `pipeline/datakit.py`. Replace placeholder private
paths with actual inputs. Source directories contain direct-child final `*.json` bundles, not author-request logs.

```bash
python3 -B -m data_pipeline inventory \
  --source claude=/mnt/f/rwkv-pi-data/datasets/private-claude/inbox \
  --source deepseek=/mnt/f/rwkv-pi-data/datasets/private-deepseek/bundles \
  --heldout-manifest /mnt/f/rwkv-pi-data/datasets/eval-holdout-v1/manifest.json \
  --out /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/inventory-NEW

python3 -B -m data_pipeline status \
  /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/inventory-NEW/catalog.json

python3 -B -m data_pipeline review-template \
  --catalog /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/inventory-NEW/catalog.json \
  --id QUESTION_ID \
  --out /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/reviews/batch-NEW.json
```

Review the prompt, input files, reference controls, and checker source for each question. Fill `reviewer`, specific
`notes`, all six review checks, and `verdict`. `calibrationReceipt` is the absolute path to an actual per-question
82 calibration JSON; `calibrationSha256` binds its bytes. It must identify the same task/hash, host, and real control
results. This module checks evidence consistency; it cannot authenticate maliciously fabricated
execution evidence.

```bash
python3 -B -m data_pipeline prepare-release \
  --catalog /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/inventory-NEW/catalog.json \
  --reviews /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/reviews/batch-NEW.json \
  --out /mnt/f/rwkv-pi-data/datasets/data-pipeline-v1/releases/batch-NEW

python3 -B -m unittest data_pipeline.test_pipeline -v
```

Outputs require mounted F:. Snapshots, review files, and release directories reject implicit overwrite. Incomplete
inventories record errors and block release. Test fixtures stay outside real catalogs and training data.

## Production quality and privacy

The user requested distinct task mechanisms and realistic environments, with substantive diversity rather than
renamed files or padded tool calls. Multi-step work means real observations and coordinated actions; meaningful
followups extend the same task/session. Difficulty is assessed after execution, including at least 20 effective calls
for the current harder-task product. Historical L1/L2 labels alone do not establish this new threshold.

Real collection uses native Pi, complete system context, Chinese questions/constraints, and real observations.
DeepSeek Flash remains the default answer teacher with high Chinese reasoning and `zh-reasoning-v2`; explicit other
teacher profiles use separate output files. The current export path uses the XML+JSON renderer/tokenizer and independent
review. Reported production metrics include usable reviewed output and throughput.

Paid requests go through the existing request ledger and owner lock; requests with unknown outcomes are never
replayed. Generators, authored questions, references, acceptance contents, trajectories,
manifests, reports, and historical backups stay on private F: storage. GitHub contains generic code and independent
engineering tests only. The user decides any later Hugging Face publication; there is no automatic upload.
