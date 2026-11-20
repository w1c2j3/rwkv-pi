/**
 * Account-style commands of the rwkv-pi launcher:
 *
 *   rwkv-pi login [TOKEN]   paste a connection token (prompted, not echoed, when omitted); verified before saving
 *   rwkv-pi status          show the saved connection and check it
 *   rwkv-pi logout          forget the saved connection
 *   rwkv-pi token --url URL (--key KEY | --key-file PATH) [--model ID] [--adapter-id ID [--adapter-version V]
 *                 [--adapter-scale S]] [--context-window N] [--max-tokens N] [--name TEXT]
 *                           issue a connection token (administrators)
 *
 * A first launch with neither a profile nor RWKV_BASE_URL runs `login` interactively.
 */
import { readFileSync } from "node:fs";
import { createInterface } from "node:readline/promises";
import { parseArgs } from "node:util";
import { decodeToken, encodeToken, profilePath, readProfile, removeProfile, verifyProfile, writeProfile, type Profile } from "./profile.js";

export const ACCOUNT_COMMANDS = ["login", "logout", "status", "token"] as const;

/** Reads one line from the terminal without echoing it. */
async function secret(prompt: string): Promise<string> {
  const rl = createInterface({ input: process.stdin, output: process.stdout, terminal: true });
  const write = (rl as unknown as { _writeToOutput: (s: string) => void })._writeToOutput.bind(rl);
  let muted = false;
  (rl as unknown as { _writeToOutput: (s: string) => void })._writeToOutput = (s: string) => { if (!muted || s.includes("\n")) write(muted ? "\n" : s); };
  const answer = rl.question(prompt);
  muted = true;
  try { return (await answer).trim(); } finally { rl.close(); }
}

const describe = (p: Profile) =>
  `${p.name ? p.name + " — " : ""}${p.url}, model ${p.model ?? "rwkv7"}${p.adapter ? `, adapter ${p.adapter.id}` : ""}`;

export async function login(token?: string, fetcher: typeof fetch = fetch): Promise<Profile> {
  if (!token) {
    if (!process.stdin.isTTY) throw new Error("No connection configured. Run: rwkv-pi login <token>");
    console.log("rwkv-pi needs a connection to an RWKV engine.\nPaste the connection token you were given (it starts with rwkvpi1.; input is hidden).");
    token = await secret("Token: ");
  }
  const profile = decodeToken(token);
  const models = await verifyProfile(profile, fetcher);
  if (profile.model && models.length && !models.includes(profile.model))
    throw new Error(`The server serves ${models.join(", ")}, not ${profile.model}; ask for a new token.`);
  const path = writeProfile(profile);
  console.log(`Connected: ${describe(profile)}\nSaved to ${path}`);
  return profile;
}

export async function runAccountCommand(command: string, args: string[]): Promise<void> {
  if (command === "login") { await login(args[0]); return; }
  if (command === "logout") { console.log(removeProfile() ? `Removed ${profilePath()}` : "No saved connection."); return; }
  if (command === "status") {
    const profile = readProfile();
    if (!profile) { console.log("No saved connection. Run: rwkv-pi login"); return; }
    console.log(`Connection: ${describe(profile)}`);
    const models = await verifyProfile(profile);
    console.log(`Engine reachable; serves ${models.join(", ") || "(unnamed)"}`);
    return;
  }
  const { values: v } = parseArgs({ args, options: {
    url: { type: "string" }, key: { type: "string" }, "key-file": { type: "string" }, model: { type: "string" },
    "adapter-id": { type: "string" }, "adapter-version": { type: "string" }, "adapter-scale": { type: "string" },
    "context-window": { type: "string" }, "max-tokens": { type: "string" }, name: { type: "string" },
  } });
  const key = v.key ?? (v["key-file"] ? readFileSync(v["key-file"], "utf8").trim() : undefined);
  if (!v.url || !key) throw new Error("rwkv-pi token needs --url and --key or --key-file.");
  const int = (s?: string) => (s === undefined ? undefined : Number(s));
  console.log(encodeToken({
    url: v.url, key, model: v.model, name: v.name,
    contextWindow: int(v["context-window"]), maxTokens: int(v["max-tokens"]),
    adapter: v["adapter-id"] ? { id: v["adapter-id"], version: v["adapter-version"], scale: int(v["adapter-scale"]) } : undefined,
  }));
}
