# Tool-call protocol

RWKV only continues text. The client renders Pi's messages and tool schemas into one prompt, streams the
continuation from `/v1/batch/completions`, and parses tool calls out of the generated text. `src/prompt.ts` is the
single renderer for both online prompts and SFT packing. Its version string is `rwkv-pi-xml-continuation-v3`.

## Prompt shape (XML format)

```text
System: <Pi system prompt>
Tools:
[<Pi tool definitions as JSON>]
When using tools, output one XML envelope per call: <tool_call>{"name":"read","arguments":{"path":"input.txt"}}</tool_call>. ...

User: <task>

Assistant: <think><reasoning></think>
<tool_call>{"name":"read","arguments":{"path":"input.txt"}}</tool_call>

User: <tool_response>
<tool output>
</tool_response>

Assistant: <think
```

- With thinking on (OpenThink), generation starts at `Assistant: <think`. With thinking off (FakeThink), it starts
  at `Assistant: <think></think>\n`.
- Past assistant turns keep their real reasoning inside `<think>…</think>`.
- A turn may contain several calls. Each call has its own envelope, and `<tool_calls>[…]</tool_calls>` batches are
  also accepted.
- The renderer controls exact bytes and whitespace. Never rebuild prompts by hand from these examples.

`RWKV_TOOL_FORMAT=json` uses native Pi `toolCall` objects instead
(`{"type":"toolCall","id":…,"name":…,"arguments":…}`) and reports tool results as JSON with `role: "tool"`. Its
version string is `rwkv-pi-json-continuation-v1`.

## Parsing rules

The parser (`src/xml.ts`) is strict by design:

- It requires a complete response and closing tags. A known tool, strict JSON, and arguments that validate against
  the tool's JSON Schema are also required.
- It rejects duplicate keys, non-finite numbers, truncated calls, and text after the first envelope.
- Examples inside Markdown fences, or written inline, stay ordinary text.
- Malformed calls are never repaired. With `RWKV_TOOL_CALL_RECOVERY=1`, a call whose envelope is complete but whose
  JSON, name, or arguments are invalid goes to Pi verbatim. Pi then returns its normal validation error to the model.
- Each response is limited to 16 calls and 1 MiB of call content.

## Stops and stream completion

- The engine stops on `<EOD>`, `✿`, `\nUser:`, and `\n### User`. The client strips the matched suffix using the
  engine's `matched_stop`.
- Tools run only after a normal finish with a configured stop and the terminal `[DONE]` event. A cancelled or
  incomplete stream never authorizes a tool call.
- Cancellation stops only this request, using its request ID. The client never issues the engine's global stop.

## Loop guards

- `RWKV_REPEAT_FAILURE_LIMIT` (default 3) ends the turn after that many identical failing calls, meaning the same
  tool, arguments, and error.
- `RWKV_IDENTICAL_CALL_LIMIT` (default 2) blocks a call after that many identical calls with identical successful
  results. Pi records the block as the call's error result.
- `RWKV_TARGET_FAILURE_LIMIT` (default 3) blocks a path-based call after that many consecutive failures of the same
  tool on the same path, whatever the other arguments were. This catches an edit retried with slightly different,
  still wrong `oldText`. Any other tool result, such as a read of the file, resets it.
- Changed arguments or changed results reset the first two counters. These guards only detect repetition; they
  don't check whether the work is correct.

## Recall-tolerant tools

With `RWKV_TOOL_TOLERANCE=1` (the default), the RWKV provider runs `read`, `edit`, `write`, and `ls` with tolerant
execution. Names, descriptions, parameters, and prompt text are pi's own, so the request the model sees is
byte-identical to the one in the training data. Only tool results change, and every tolerated deviation is stated in
the result:

- **Paths.** A missing path resolves to the one workspace file it plausibly means: the longest existing suffix,
  collapsed repeated directories, or a unique basename. Otherwise the error lists candidate paths.
- **Edits.** When pi's own matching fails, `oldText` is matched in order ignoring indentation, then blank lines, then
  whitespace inside lines, then as a unique close match of at least 90% similarity. A level with several matches
  stops as ambiguous instead of trying a looser level. The replacement is re-indented to the file's indentation,
  and recall drift in context never reaches the file: lines the edit leaves unchanged (identical in `oldText` and
  `newText`) are written as the file has them, and blank lines the old text left out stay in place. A one-line
  `oldText` that pi would match inside a line's indentation is widened to the whole line first, so multi-line
  replacements are not under-indented. A failed edit reports the closest file text to copy from, or, when the new
  text already stands where the old text belongs, that the change appears applied.
- **Writes.** Overwriting a file reports how many lines changed and which previous lines are gone, and new
  directories are named.

Teacher runs use pi's tools unchanged.

## Context and training alignment

- Prompts are measured with the engine's `/v1/tokens/count`. When space runs short, the output allowance shrinks
  instead of truncating input.
- The full history is sent on every request, so the engine's prefix cache can reuse a matching snapshot.
- A tool-only assistant turn is rendered so that it extends the previous generation prompt exactly. This preserves
  prefix reuse.
- Packing (`pipeline/compile-lightning-dataset.mjs`) uses the same renderer and the native tokenizer. Context
  segments are not supervised. Complete assistant targets, including EOS token 0, follow the
  `segment-end-target-v1` contract, and the packer checks token prefixes and segment concatenation.
