/**
 * Saved connection profile, the zero-configuration path for colleagues: an administrator issues a connection token
 * (`rwkv-pi token`), the user pastes it once (`rwkv-pi login`), and every later launch reads the profile.
 * Environment variables and .env files still take precedence, so existing setups are unchanged.
 *
 * Token format: "rwkvpi1." + base64url(JSON profile). It carries the engine credential, so it is a secret.
 */
import { chmodSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

export const TOKEN_PREFIX = "rwkvpi1.";

export interface Profile {
  /** Engine API root, ending in /v1. */
  url: string;
  key: string;
  model?: string;
  adapter?: { id: string; version?: string; scale?: number };
  contextWindow?: number;
  maxTokens?: number;
  /** Display name, e.g. the hosting server. */
  name?: string;
}

export function profilePath(env: NodeJS.ProcessEnv = process.env): string {
  const dir = env.RWKV_PI_CONFIG_DIR?.trim() || join(env.XDG_CONFIG_HOME?.trim() || join(homedir(), ".config"), "rwkv-pi");
  return join(dir, "profile.json");
}

function validate(value: unknown): Profile {
  const p = value as Profile;
  const fail = (why: string): never => { throw new Error(`Invalid connection profile: ${why}.`); };
  if (!p || typeof p !== "object") fail("not an object");
  let url: URL | undefined;
  try { url = new URL(p.url); } catch { fail("url is not an absolute URL"); }
  if (!["http:", "https:"].includes(url!.protocol) || url!.username || url!.password || url!.search || url!.hash) fail("url must be plain http(s)");
  if (typeof p.key !== "string" || !p.key) fail("key is missing");
  if (p.model !== undefined && (typeof p.model !== "string" || !p.model)) fail("model must be a non-empty string");
  if (p.adapter !== undefined && (typeof p.adapter?.id !== "string" || !p.adapter.id)) fail("adapter.id is missing");
  if (p.adapter?.scale !== undefined && !Number.isFinite(p.adapter.scale)) fail("adapter.scale must be finite");
  for (const k of ["contextWindow", "maxTokens"] as const)
    if (p[k] !== undefined && !(Number.isSafeInteger(p[k]) && p[k]! > 0)) fail(`${k} must be a positive integer`);
  return { ...p, url: p.url.replace(/\/+$/, "") };
}

export function encodeToken(profile: Profile): string {
  return TOKEN_PREFIX + Buffer.from(JSON.stringify(validate(profile))).toString("base64url");
}

export function decodeToken(token: string): Profile {
  const t = token.trim();
  if (!t.startsWith(TOKEN_PREFIX)) throw new Error(`A connection token starts with ${TOKEN_PREFIX}`);
  let json: unknown;
  try { json = JSON.parse(Buffer.from(t.slice(TOKEN_PREFIX.length), "base64url").toString("utf8")); }
  catch { throw new Error("The connection token is damaged; copy it again in full."); }
  return validate(json);
}

export function readProfile(env: NodeJS.ProcessEnv = process.env): Profile | undefined {
  const path = profilePath(env);
  return existsSync(path) ? validate(JSON.parse(readFileSync(path, "utf8"))) : undefined;
}

export function writeProfile(profile: Profile, env: NodeJS.ProcessEnv = process.env): string {
  const path = profilePath(env);
  mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
  writeFileSync(path, JSON.stringify(validate(profile), null, 2) + "\n", { mode: 0o600 });
  chmodSync(path, 0o600);
  return path;
}

export function removeProfile(env: NodeJS.ProcessEnv = process.env): boolean {
  const path = profilePath(env);
  if (!existsSync(path)) return false;
  rmSync(path);
  return true;
}

/** True when the environment already names an engine, so no profile is needed. */
export function configuredByEnv(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.RWKV_BASE_URL?.trim());
}

/** Fills RWKV_* variables from the profile, never overriding a variable that is already set. */
export function applyProfile(profile: Profile, env: NodeJS.ProcessEnv = process.env): void {
  const set = (name: string, value: string | number | undefined) => {
    if (value !== undefined && !env[name]?.trim()) env[name] = String(value);
  };
  set("RWKV_BASE_URL", profile.url);
  if (!env.RWKV_API_KEY?.trim() && !env.RWKV_API_KEY_FILE?.trim()) env.RWKV_API_KEY = profile.key;
  set("RWKV_MODEL", profile.model);
  set("RWKV_CONTEXT_WINDOW", profile.contextWindow);
  set("RWKV_MAX_TOKENS", profile.maxTokens);
  if (profile.adapter && !env.RWKV_ADAPTER_ID?.trim()) {
    env.RWKV_ADAPTER_ID = profile.adapter.id;
    set("RWKV_ADAPTER_VERSION", profile.adapter.version);
    set("RWKV_ADAPTER_SCALE", profile.adapter.scale);
  }
}

/** Checks the engine accepts the credential; returns the served model ids. */
export async function verifyProfile(profile: Profile, fetcher: typeof fetch = fetch): Promise<string[]> {
  let response: Response;
  try {
    response = await fetcher(profile.url + "/models", { headers: { Authorization: `Bearer ${profile.key}` }, signal: AbortSignal.timeout(15000) });
  } catch (error) {
    throw new Error(`Cannot reach ${profile.url}: ${(error as Error).message}. Check the network or VPN.`);
  }
  if (response.status === 401 || response.status === 403) throw new Error("The server rejected this token; ask for a new one.");
  if (!response.ok) throw new Error(`${profile.url}/models answered HTTP ${response.status}.`);
  return ((await response.json() as { data?: { id?: string }[] }).data ?? []).map(m => m.id ?? "").filter(Boolean);
}
