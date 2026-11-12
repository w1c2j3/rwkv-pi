# rwkv-pi

RWKV integration for the Pi coding agent (`@earendil-works/pi-coding-agent`). Pi keeps its real agent loop,
tools, sessions, and terminal interface. RWKV runs on the
[`rwkv_lightning_cuda`](https://github.com/Alic-Li/rwkv_lightning_cuda) engine as a plain text-continuation model, and
this client translates between the two:

```text
Pi messages + tools + observations
               |
      shared client renderer          src/prompt.ts
               |
   Lightning continuation (SSE)       src/lightning.ts
               |
 validated text and tool calls        src/xml.ts
               |
     real Pi tools -> next turn
```

The same renderer and the engine's native tokenizer are used to pack training data, so online prompts and SFT rows
are byte-identical.

## Requirements

- Node.js 22.19 or newer.
- A running `rwkv_lightning_cuda` server. The pinned build is described in `lightning-source.lock.json`.
- Linux with bubblewrap for isolated harness runs and sandboxed subagents.
- git and g++ (the installer builds the pinned Lightning tokenizer from source).
- Python 3 for the optional engine installer and the Python parser port in `inference/`.

## Quick start

```bash
npm run setup                    # = scripts/install.sh: npm ci, patched Pi source, pinned Lightning tokenizer, .env, offline doctor
$EDITOR .env                     # point RWKV_BASE_URL / RWKV_MODEL at your rwkv_lightning_cuda engine
npm run doctor                   # one-screen check, including the live engine and tokenizer parity
npm run start:source -- --offline --thinking off
```

Pass `--thinking high` to enable reasoning. `source scripts/activate-pi.sh` exposes the pinned launcher as `pi`
in the current shell. Exported variables override values in `.env`.

`npm run doctor` checks Node, the pinned Pi commit and its applied patches, the Lightning-native tokenizer build and
vocabulary hash, configuration, the data root, bubblewrap and, unless `--offline`, the engine's `/v1/models` and
`/v1/tokens/count`. When both are available it requires the offline tokenizer and the engine to agree on a probe,
so packed training data and served prompts tokenize identically. It exits non-zero on any failure.

### Tokenizer

RWKV token counts come only from the Lightning-native tokenizer: online through the engine's `/v1/tokens/count`, and
offline (dataset packing) through the same upstream `trie.h` built by `npm run lightning:tokenizer` into
`.upstream/lightning-tokenizer/`. No Hugging Face, irwkv or vLLM tokenizer is used anywhere in the RWKV path.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RWKV_BACKEND` | `lightning` | `lightning` for RWKV. `openai` exists only for teacher models and test mocks; it has no RWKV token budget |
| `RWKV_BASE_URL` | `http://127.0.0.1:8000/v1` | Engine API root |
| `RWKV_MODEL` | `rwkv7` | Model identity reported to Pi |
| `RWKV_API_KEY` / `RWKV_API_KEY_FILE` | `rwkv-local` | Engine credential |
| `RWKV_CONTEXT_WINDOW` | `131072` | Total context budget (prompt + output) |
| `RWKV_MAX_TOKENS` | `8192` | Output budget; must be smaller than the context window |
| `RWKV_TOOL_FORMAT` | `xml` | `xml` envelopes or native Pi `json` tool calls (see [protocol](docs/protocol.md)) |
| `RWKV_PROMPT_TEMPLATE` | matches the format | Must equal the tool format on Lightning |
| `RWKV_TOOLS` | `read,write,edit,bash` | Pi tools offered; `grep`, `find`, `ls` are optional |
| `RWKV_REPEAT_FAILURE_LIMIT` | `3` | Stop after this many identical failing calls (`0` disables) |
| `RWKV_IDENTICAL_CALL_LIMIT` | `2` | Block a call repeated with identical successful results (`0` disables) |
| `RWKV_TARGET_FAILURE_LIMIT` | `3` | Block a tool on a path after this many consecutive failures on it (`0` disables) |
| `RWKV_TOOL_TOLERANCE` | `1` | Recall-tolerant read/edit/write execution for the RWKV provider (`0` uses pi's tools unchanged) |
| `RWKV_TOOL_CALL_RECOVERY` | `1` | Hand malformed but complete calls to Pi's validator instead of aborting |
| `RWKV_ADAPTER_ID`, `_VERSION`, `_SCALE` | unset | Optional adapter selection on the engine |
| `RWKV_PLUGINS` | `none` | `strong-model`, `subagents`, or both, comma-separated |
| `RWKV_PI_DATA_ROOT` | `$XDG_DATA_HOME/rwkv-pi` (`~/.local/share/rwkv-pi`) | Collaboration traces and participation records. An explicitly set root must exist; it never falls back to another disk. |

Teacher, subagent, and stall-detection settings are listed in [workflows](docs/workflows.md).
Input is never silently truncated: `/v1/tokens/count` measures the prompt, and the request fails if the
context is exhausted.

## Commands

| Command | What it does |
| --- | --- |
| `npm run start:source` | Interactive Pi with the RWKV provider (`--plugins …` to enable workflows) |
| `npm run orchestrate` | Teacher main agent that plans and reviews; RWKV subagents execute |
| `npm run step` | Pi with the optional StepFun teacher provider |
| `npm run harness -- --tasks T.jsonl --out DIR --backend rwkv` | Isolated batch runs with recorded requests and checks |
| `npm run collab -- run --task T.jsonl` | RWKV first, teacher takeover on stall, with a comparison report |
| `npm run dataset -- --runs DIR --out DIR` | Extract chat records from harness runs |
| `npm run tokenize -- --dataset DIR --out DIR` | Pack records with the shared renderer and native tokenizer |
| `npm run lightning:setup` | Clone the pinned Lightning source and build its native CPU tokenizer |

See [harness and datasets](docs/harness.md) for the run and packing formats.

## Engine installation

The CUDA engine is installed from the exact GitHub Actions artifact recorded in `lightning-source.lock.json`.
The installer verifies the outer and inner digests before extracting:

```bash
python3 scripts/install-lightning.py --archive /path/to/package.zip --out /path/to/lightning-engine
```

The native executable is spelled `rwkv_lighting_cuda` upstream. Deployment identity comes from the lock files,
not from a similarly named release.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Provider, renderer, parser, harness, sandbox, collaboration and subagent extensions, CLIs |
| `test/` | Node test suites with a mock engine (`test/helpers/mock-server.mjs`) |
| `inference/` | Python port of the XML parser (`xml_protocol.py`) for Python-side QC, with tests |
| `pipeline/` | Native-tokenizer packing (`compile-lightning-dataset.mjs`, `trajectory-packing.mjs`) |
| `scripts/` | Pi/engine setup, tokenizer build, smoke run |
| `patches/` | Reviewed patches applied to the pinned Pi and Lightning sources |
| `templates/` | `rwkv-xml-v1` Jinja chat template for serving, derived from the RWKV7-G1l template (Apache-2.0, see `templates/LICENSE`) |
| `tasks/smoke.jsonl` | Synthetic smoke tasks |

## Development

```bash
npm run check          # type-check
npm test               # build, then the Node suites (sandbox-search skips unless Pi is patched)
npm run test:source    # the suites that need the patched Pi source; REQUIRE_SANDBOX_TESTS=1 makes skips fail
npm run smoke          # end-to-end smoke tasks against the mock engine
python3 -m unittest discover -s test -p 'test_*.py'
```

Test results show that the software behaves correctly. They are not evidence of model capability.
