import { Ajv } from "ajv";
import { parseTree, getNodeValue, type Node, type ParseError } from "jsonc-parser";
import { randomUUID } from "node:crypto";
import { STOP_TEXTS, type ToolFormat } from "./prompt.js";

const ajv = new Ajv({ strict: false, allErrors: false });
const stopLookbehind = Math.max(...STOP_TEXTS.map(stop => stop.length));
function strictJson(source: string): unknown {
  const errors: ParseError[] = [];
  const tree = parseTree(source, errors, { allowTrailingComma: false, disallowComments: true });
  if (!tree || errors.length) throw new Error("Invalid JSON in XML tool call.");
  function visit(node: Node, depth: number): void {
    if (depth > 100) throw new Error("Tool arguments exceed nesting limit.");
    if (node.type === "object") {
      const keys = node.children?.map(p => p.children?.[0]?.value) ?? [];
      if (new Set(keys).size !== keys.length) throw new Error("Duplicate JSON key.");
    }
    if (node.type === "number" && !Number.isFinite(node.value)) throw new Error("Non-finite JSON number.");
    for (const child of node.children ?? []) visit(child, depth + 1);
  }
  visit(tree, 0);
  return getNodeValue(tree);
}

export function callStart(text: string, partial = false): number | undefined {
  let offset = 0, fence: string | undefined;
  for (const line of text.match(/[^\n]*\n|[^\n]+$/g) ?? []) {
    const clean = line.replace(/^[ \t]*/, "");
    const marker = ["```", "~~~"].find(m => clean.startsWith(m));
    if (marker) fence = fence === marker ? undefined : fence ?? marker;
    else if (!fence && ["<tool_call>", "<tool_calls>"].some(tag => clean.startsWith(tag) || (partial && tag.startsWith(clean) && clean.length))) return offset + line.length - clean.length;
    offset += line.length;
  }
  return undefined;
}

/** Parse complete frames only. JSON string contents may themselves contain closing tags.
 * recover=false: any malformed frame aborts the turn (strict, rwkv-xml-continuation-v2).
 * recover=true: an envelope whose JSON, tool name or arguments are invalid is still handed to pi with
 * the model's own name and argument text, so pi's native validation returns a tool error result the
 * model can react to (unknown tool -> "Tool X not found"; bad JSON -> "Invalid arguments ..."; schema
 * mismatch -> "Validation failed ..." after pi's standard coercion). Nothing is invented: arguments are
 * the generated text, never repaired. Text outside envelopes and unnamed frames remain fatal. */
export function parseCalls(source: string, tools: any[], choice: any = "auto", recover = false) {
  if (Buffer.byteLength(source) > 1024 * 1024) throw new Error("XML calls exceed 1 MiB.");
  const allowed = new Map(tools.map(t => [t.function.name, t.function.parameters]));
  const calls: any[] = [];
  const push = (name: string, args: string) => {
    if (choice === "none") throw new Error("Tool is not allowed.");
    if (typeof choice === "object" && choice.function?.name !== name) throw new Error("Named tool choice mismatch.");
    calls.push({ index: calls.length, id: `call_${randomUUID().replaceAll("-", "")}`, type: "function", function: { name, arguments: args } });
    if (calls.length > 16) throw new Error("Too many tool calls.");
  };
  let tail = source.trim();
  while (tail) {
    const opening = ["<tool_call>", "<tool_calls>"].find(t => tail.startsWith(t));
    if (!opening) throw new Error("Expected only XML tool-call envelopes.");
    tail = tail.slice(opening.length).trimStart();
    const closing = opening.replace("<", "</");
    // Locate a JSON boundary without treating closing tags inside strings as delimiters.
    let quoted = false, escape = false, depth = 0, end = -1;
    for (let i = 0; i < tail.length; i++) {
      const c = tail[i];
      if (quoted) { if (escape) escape = false; else if (c === "\\") escape = true; else if (c === '"') quoted = false; }
      else if (c === '"') quoted = true;
      else if (c === "{" || c === "[") depth++;
      else if (c === "}" || c === "]") { if (--depth === 0) { end = i + 1; break; } }
    }
    if (end < 0 && !recover) throw new Error("Incomplete XML JSON payload.");
    let value: any;
    try { if (end > 0) value = strictJson(tail.slice(0, end)); } catch (error) { if (!recover) throw error; }
    const after = tail.slice(Math.max(end, 0)).trimStart();
    if (value !== undefined && after.startsWith(closing)) tail = after.slice(closing.length).trim();
    else if (value !== undefined && recover && !after) tail = ""; // closing tag missing at the end of the turn
    else {
      if (!recover) throw new Error("Missing XML closing tag.");
      // Broken JSON or extra braces: the frame runs to the next closing tag (or the end of the turn)
      // and its argument text is handed to pi verbatim.
      const close = tail.indexOf(closing, Math.max(end, 0));
      const frame = (close < 0 ? tail : tail.slice(0, close)).trim();
      tail = close < 0 ? "" : tail.slice(close + closing.length).trim();
      const head = /^\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"arguments"\s*:\s*/.exec(frame);
      if (!head) throw new Error("Invalid JSON in XML tool call.");
      let raw = frame.slice(head[0].length).trim();
      if (raw.endsWith("}")) raw = raw.slice(0, -1).trimEnd(); // the envelope object's own brace
      push(JSON.parse(`"${head[1]}"`), raw);
      continue;
    }
    const batch = opening === "<tool_calls>" ? value : [value];
    if (!Array.isArray(batch) || !batch.length) throw new Error("Empty tool-call frame.");
    for (let f of batch) {
      if (f?.function) {
        if (!recover && (Object.keys(f).some(k => !["type", "id", "function"].includes(k)) || (f.type && f.type !== "function"))) throw new Error("Invalid function wrapper.");
        f = f.function;
      }
      if (recover) {
        if (!f || typeof f !== "object" || typeof f.name !== "string" || !f.name) throw new Error("Tool is not allowed.");
        push(f.name, typeof f.arguments === "string" ? f.arguments : JSON.stringify(f.arguments ?? {}));
        continue;
      }
      if (!f || Object.keys(f).sort().join() !== "arguments,name" || !allowed.has(f.name) || choice === "none") throw new Error("Tool is not allowed.");
      const args = typeof f.arguments === "string" ? strictJson(f.arguments) : f.arguments;
      if (!args || typeof args !== "object" || Array.isArray(args) || !ajv.validate(allowed.get(f.name), args)) throw new Error(`Invalid arguments for ${f.name}.`);
      push(f.name, JSON.stringify(args));
    }
  }
  if (!calls.length) throw new Error("No tool calls.");
  return calls;
}

/** Native pi ToolCall JSON. Preserve model/capture IDs; never invent arguments. */
export function parseJsonCalls(source: string, tools: any[], choice: any = "auto") {
  if (Buffer.byteLength(source) > 1024 * 1024) throw new Error("JSON calls exceed 1 MiB.");
  const value: any = strictJson(source.trim());
  const batch = Array.isArray(value) ? value : [value];
  const allowed = new Map(tools.map(t => [t.function.name, t.function.parameters]));
  const looksLikeCall = batch.some(f => f && typeof f === "object"
    && (f.type === "toolCall" || Object.hasOwn(f, "arguments") || allowed.has(f.name)));
  if (!looksLikeCall) return []; // Ordinary JSON answers remain text.
  if (!batch.length || batch.length > 16) throw new Error("Invalid JSON tool call count.");
  const ids = new Set<string>();
  return batch.map((f, index) => {
    if (!f || Object.keys(f).sort().join() !== "arguments,id,name,type" || f.type !== "toolCall"
      || typeof f.id !== "string" || !/^[a-zA-Z0-9_-]{1,128}$/.test(f.id) || ids.has(f.id)) throw new Error("Invalid native pi toolCall identity.");
    if (!allowed.has(f.name) || choice === "none" || (typeof choice === "object" && choice.function?.name !== f.name)) throw new Error("Tool is not allowed.");
    if (!f.arguments || typeof f.arguments !== "object" || Array.isArray(f.arguments)
      || !ajv.validate(allowed.get(f.name), f.arguments)) throw new Error(`Invalid arguments for ${f.name}.`);
    ids.add(f.id);
    return { index, id: f.id, type: "function", function: { name: f.name, arguments: JSON.stringify(f.arguments) } };
  });
}

export function jsonCallStart(text: string): number | undefined {
  let offset = 0, fence: string | undefined;
  for (const line of text.match(/[^\n]*\n|[^\n]+$/g) ?? []) {
    const clean = line.replace(/^[ \t]*/, "");
    const marker = ["```", "~~~"].find(m => clean.startsWith(m));
    if (marker) fence = fence === marker ? undefined : fence ?? marker;
    else if (!fence && (clean.startsWith("{") || clean.startsWith("["))) return offset + line.length - clean.length;
    offset += line.length;
  }
  return undefined;
}

export class ContinuationParser {
  raw = "";
  private reasonSent = 0;
  private textSent = 0;
  constructor(private thinking: boolean, private tools: any[], private choice: any = "auto", private format: ToolFormat = "xml", private recover = false) {}
  private callStart(text: string) { return this.format === "json" ? jsonCallStart(text) : callStart(text, true); }
  private split() {
    if (!this.thinking) return { reasoning: "", text: this.raw, open: false };
    const full = "<think" + this.raw;
    const start = full.indexOf(">");
    const end = full.indexOf("</think>");
    return { reasoning: start < 0 ? "" : full.slice(start + 1, end < 0 ? Math.max(start + 1, full.length - 8) : end), text: end < 0 ? "" : full.slice(end + 8), open: end < 0 };
  }
  push(chunk: string): Record<string, unknown>[] {
    this.raw += chunk;
    if (Buffer.byteLength(this.raw) > 4 * 1024 * 1024) throw new Error("Continuation exceeds 4 MiB.");
    const { reasoning, text } = this.split();
    const deltas: Record<string, unknown>[] = [];
    if (reasoning.length > this.reasonSent) deltas.push({ reasoning_content: reasoning.slice(this.reasonSent) });
    this.reasonSent = reasoning.length;
    const start = this.callStart(text);
    // Native text stops match a generated suffix at token boundaries. Hold that
    // suffix until finish metadata arrives, so stop markers never reach pi.
    const safe = start ?? Math.max(0, text.length - stopLookbehind);
    if (safe > this.textSent) { deltas.push({ content: text.slice(this.textSent, safe) }); this.textSent = safe; }
    return deltas;
  }
  finish(reason: string, matchedStop?: string): { deltas: Record<string, unknown>[]; reason: string } {
    if (matchedStop !== undefined) {
      if (reason !== "stop" || !STOP_TEXTS.includes(matchedStop) || !this.raw.endsWith(matchedStop)) throw new Error("Invalid native stop metadata.");
      this.raw = this.raw.slice(0, -matchedStop.length);
    }
    const { text, open } = this.split();
    const start = this.callStart(text);
    if (open && reason !== "length") throw new Error("Unclosed reasoning block.");
    if (start !== undefined && reason !== "stop") throw new Error("Truncated tool call; execution refused.");
    const calls = start === undefined ? [] : (this.format === "json" ? parseJsonCalls(text.slice(start), this.tools, this.choice) : parseCalls(text.slice(start), this.tools, this.choice, this.recover));
    if (!calls.length && (this.choice === "required" || typeof this.choice === "object") && reason === "stop") throw new Error("Required tool call missing.");
    const deltas: Record<string, unknown>[] = [];
    if (open) {
      const full = "<think" + this.raw, pos = full.indexOf(">");
      if (pos >= 0) deltas.push({ reasoning_content: full.slice(pos + 1 + this.reasonSent) });
    }
    const end = calls.length ? start! : text.length;
    if (end > this.textSent) deltas.push({ content: text.slice(this.textSent, end) });
    if (calls.length) deltas.push({ tool_calls: calls });
    return { deltas, reason: calls.length ? "tool_calls" : reason };
  }
}
