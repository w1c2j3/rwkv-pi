# rwkv-pi: code-only GitHub policy

- User instruction (2026-11-08): no questions, fixtures used as real training tasks, references, task-specific acceptance checks, question generators, trajectories, datasets, evaluation reports or copied private histories on GitHub, including private repositories and commit history. Only generic application/pipeline code belongs here.
- Data stays on mounted `/mnt/f/rwkv-pi-data`. Keep all expensive authored assets and original local working files. After efficacy verification, the user decides what to publish on Hugging Face. No automatic upload to either service.
- `.github/code-only-files.txt` is the explicit reviewed path allowlist. Review contents when adding or changing entries; do not weaken it to pass a push. The hooks and CI inspect complete trees/history, not just new file names. They are guardrails, not proof against every possible disguised disclosure.
- Never merge the retired private history back into `main` or `data`. The offline Git bundle and working-file backup stay on F:. Never push backup refs, old tags, history bundles or sensitive logs.
- Keep `main` as default and literal `data` for data-pipeline development. Preserve original working files in the original checkout. No training or 157 service changes without an explicit user instruction.
- The answer teacher remains DeepSeek Flash with high Chinese reasoning, complete native Pi system prompt and existing Chinese constraints, `zh-reasoning-v2`, canonical XML+JSON. Existing private budgets/unknown reservations remain authoritative; this change authorizes no paid calls.
- Engineering fixtures must remain obviously synthetic and independent of private questions. Never use real task bodies as tests. Do not include task contents in CI output.
