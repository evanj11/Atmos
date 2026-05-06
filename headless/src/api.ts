// ─── api.ts ───────────────────────────────────────────────────────────────────
//
// Drop-in replacement for `import { invoke } from '@tauri-apps/api/core'`.
// Every call becomes POST /api/<command> with a JSON body.
// The server responds with { ok: true, payload: T } or { ok: false, error: string }.

export async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const res = await fetch(`/api/${command}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(args ?? {}),
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status} from /api/${command}`);
  }

  const json = await res.json();
  if (!json.ok) {
    throw new Error(json.error ?? `Unknown server error from /api/${command}`);
  }
  return json.payload as T;
}
