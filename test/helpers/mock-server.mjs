import { once } from "node:events";
import { createServer } from "node:http";

export function toolCall(name, args, id = "call_smoke") {
  return { tool_calls: [{ index: 0, id, type: "function", function: { name, arguments: JSON.stringify(args) } }] };
}

export async function mockServer(respond) {
  const requests = [];
  const server = createServer(async (req, res) => {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    const body = JSON.parse(raw);
    requests.push(body);
    const result = respond(body, requests.length);
    if (result === null) return;
    res.writeHead(200, { "content-type": "text/event-stream" });
    for (const delta of result.deltas) {
      res.write(`data: ${JSON.stringify({ id: "mock", object: "chat.completion.chunk", created: 1, model: "mock-only", choices: [{ index: 0, delta, finish_reason: null }] })}\n\n`);
    }
    res.write(`data: ${JSON.stringify({ id: "mock", object: "chat.completion.chunk", created: 1, model: "mock-only", choices: [{ index: 0, delta: {}, finish_reason: result.finish ?? "stop" }] })}\n\n`);
    res.end("data: [DONE]\n\n");
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return {
    url: `http://127.0.0.1:${server.address().port}/v1`, requests,
    close: () => new Promise((resolve) => { server.close(resolve); server.closeAllConnections(); }),
  };
}

export function copyFileResponses(_body, index) {
  if (index === 1) return { deltas: [{ reasoning_content: "Read the input first." }, toolCall("read", { path: "input.txt" }, "call_read")], finish: "tool_calls" };
  if (index === 2) return { deltas: [toolCall("write", { path: "output.txt", content: "HELLO PI\n" }, "call_write")], finish: "tool_calls" };
  return { deltas: [{ content: "Wrote output.txt." }] };
}
