/** Canonical XML+JSON continuation format; shared by the client and SFT compiler. */
export const PROMPT_VERSION = "rwkv-pi-xml-continuation-v3";
export const JSON_PROMPT_VERSION = "rwkv-pi-json-continuation-v1";
export type ToolFormat = "xml" | "json";
export const promptVersion = (format: ToolFormat = "xml") => format === "json" ? JSON_PROMPT_VERSION : PROMPT_VERSION;
export function formatForPromptVersion(version: string): ToolFormat {
  if (version === PROMPT_VERSION) return "xml";
  if (version === JSON_PROMPT_VERSION) return "json";
  throw new Error(`Unknown prompt version: ${version}`);
}
// Preserve the base model's turn boundaries, including tool-result handoff.
export const STOP_TEXTS = ["<EOD>", "✿", "\nUser:", "\n### User"];
type Message = Record<string, any>;

function text(content: unknown): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content) && content.every(p => p.type === "text" && typeof p.text === "string")) return content.map(p => p.text).join("\n");
  throw new Error("RWKV continuation accepts text messages only.");
}

export function renderPrompt(messages: Message[], tools: unknown[] = [], thinking = false, generation = true, format: ToolFormat = "xml"): string {
  if (!messages.length) throw new Error("Empty conversation.");
  const system = messages.filter(m => m.role === "system").map(m => text(m.content).trim()).join("\n\n");
  let output = "";
  if (system || tools.length) {
    output += `System: ${system}`;
    if (tools.length) output += `${system ? "\n" : ""}Tools:\n${JSON.stringify(tools)}\n` + (format === "json"
      ? 'When using tools, output a native pi toolCall JSON object on its own line: {"type":"toolCall","id":"call_1","name":"read","arguments":{"path":"input.txt"}}. Use a new call id for each call. Use only the declared tool names and arguments. Do not wrap calls in XML or Markdown fences. For multiple calls output one JSON array of these objects. Tool results arrive as JSON with role tool and a matching tool_call_id. Otherwise answer normally; put illustrative tool-call JSON inside Markdown code fences.'
      : 'When using tools, output one XML envelope per call: <tool_call>{"name":"read","arguments":{"path":"input.txt"}}</tool_call>. Use only the declared tool names and arguments. Tool results arrive inside <tool_response>. Otherwise answer normally.');
    output += "\n\n";
  }
  for (const m of messages) {
    if (m.role === "system") continue;
    if (m.role === "user") output += `User: ${text(m.content).trim()}\n\n`;
    else if (m.role === "tool") output += format === "json"
      ? `User: ${JSON.stringify({ role: "tool", tool_call_id: m.tool_call_id, content: text(m.content) })}\n\n`
      : `User: <tool_response>\n${text(m.content)}\n</tool_response>\n\n`;
    else if (m.role === "assistant") {
      const reasoning = text(m.reasoning_content ?? m.reasoning);
      if (/<\/?think>/.test(reasoning)) throw new Error("Ambiguous reasoning delimiters.");
      const parts = [text(m.content).trim()].filter(Boolean);
      const jsonCalls: { type: "toolCall"; id: string; name: string; arguments: unknown }[] = [];
      for (const call of m.tool_calls ?? []) {
        const f = call.function ?? call;
        const args = typeof f.arguments === "string" ? JSON.parse(f.arguments) : f.arguments;
        if (format === "json") {
          if (typeof call.id !== "string" || !call.id) throw new Error("Native pi JSON calls require their original id.");
          jsonCalls.push({ type: "toolCall", id: call.id, name: f.name, arguments: args });
        }
        else parts.push(`<tool_call>${JSON.stringify({ name: f.name, arguments: args })}</tool_call>`);
      }
      if (jsonCalls.length) parts.push(JSON.stringify(jsonCalls.length === 1 ? jsonCalls[0] : jsonCalls));
      // A tool-only turn must extend the generation prefix verbatim. An extra
      // newline here merges its final token (11 -> 261), defeating prefix reuse.
      output += `Assistant: <think>${reasoning}</think>\n${parts.join("\n")}\n\n`;
    } else throw new Error(`Unsupported role: ${m.role}`);
  }
  if (generation) output += thinking ? "Assistant: <think" : "Assistant: <think></think>\n";
  return output;
}

export function promptForRequest(request: Record<string, any>, format: ToolFormat = "xml"): { prompt: string; thinking: boolean } {
  if (!Array.isArray(request.messages)) throw new Error("Missing chat messages.");
  const kwargs = request.chat_template_kwargs ?? {};
  const thinking = kwargs.thinking ?? kwargs.fake_cot_mode === false;
  return { prompt: renderPrompt(request.messages, request.tools ?? [], thinking, true, format), thinking };
}
