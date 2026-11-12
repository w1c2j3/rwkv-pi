# rwkv-pi

RWKV Lightning continuation client and generic data-pipeline code for the native Pi agent.
The production protocol is XML envelopes containing JSON, parsed on the client.

## Repository boundary

GitHub contains **code only**, even when the repository is private. Paid/self-authored questions,
question generators, initial task files, reference answers, task-specific acceptance checks,
model trajectories, datasets and evaluation reports stay on private storage. After effectiveness
validation, the owner may publish selected data to Hugging Face separately.

The file allowlist, local Git hooks and CI check complete commit history to prevent accidentally
reintroducing private assets. Every newly allowed file still needs content review. Enable hooks:

```bash
git config core.hooksPath .githooks
python3 -B scripts/check-code-only.py --refs HEAD
```

## Development

`main` contains the generic client baseline. `data` adds the independent `data_pipeline/` module.
Private legacy workflows remain available locally outside the published Git tree.

```bash
npm ci
npm run check
npm test
```

The committed tests use tiny engineering-only inputs, never corpus questions. Actual model execution,
acceptance, tokenization and training require separately provided private artifacts and authorization.

## Data products and documentation

See [the data pipeline guide](data_pipeline/README.md) and [the two output modes](data_pipeline/outputs.md).
Frozen questions and reviewed teacher trajectories are separate products; multiple explicitly configured teachers
can answer the same question snapshot while keeping their training files separate. DeepSeek is the default teacher.

Human-facing documentation is English. Questions, fixtures, model-facing prompts, Chinese reasoning instructions,
and training data remain in their original Chinese. Local private `CLAUDE.md` and `docs/HANDOFF.md` provide operator
context and current state; keep them ignored rather than adding them to the GitHub allowlist.
