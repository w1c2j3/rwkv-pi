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
- Python 3 for the optional engine installer and the inference-side parser in `inference/`.

## Quick start

```bash
npm ci
npm run pi:setup                 # fetch and patch the pinned Pi source (pi-source.lock.json)
cp .env.example .env             # then point RWKV_BASE_URL at your engine
npm run build
npm run start:source -- --offline --thinking off
```

Pass `--thinking high` to enable reasoning. `source scripts/activate-pi.sh` exposes the pinned launcher as `pi`
in the current shell. Exported variables override values in `.env`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RWKV_BACKEND` | `lightning` | `lightning` (client-side parsing) or `openai` (legacy native tool calls) |
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
| `RWKV_PI_DATA_ROOT` | `/mnt/f/rwkv-pi-data` | Where collaboration traces and participation records are written. Set this on every new machine. |

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
| `inference/` | Python XML tool parser for serving-side integration, with tests |
| `pipeline/` | Native-tokenizer packing (`compile-lightning-dataset.mjs`, `trajectory-packing.mjs`) |
| `scripts/` | Pi/engine setup, tokenizer build, smoke run |
| `patches/` | Reviewed patches applied to the pinned Pi and Lightning sources |
| `templates/` | `rwkv-xml-v1` Jinja chat template for serving, derived from the RWKV7-G1l template (Apache-2.0, see `templates/LICENSE`) |
| `tasks/smoke.jsonl` | Synthetic smoke tasks |

## Development

```bash
npm run check          # type-check
npm test               # build, then run the Node test suites
npm run smoke          # end-to-end smoke tasks against the mock engine
python3 -m unittest discover -s test -p 'test_*.py'
```

`npm test` also globs `pipeline/test/`. That directory belongs to the data pipeline, which this repository does not
include, so name the test files explicitly when you run without it. Test results show that the software behaves
correctly. They are not evidence of model capability.
