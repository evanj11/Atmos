// ─── events.ts ────────────────────────────────────────────────────────────────
//
// Drop-in replacement for `import { listen } from '@tauri-apps/api/event'`.
// A single WebSocket connection to /api/events receives all server-pushed
// messages.  Messages from the server are JSON objects with shape:
//
//   { event: string, payload: any }   — for viz events (viz:frame, viz:residue, …)
//   { tool: string,  pct: number }    — for progress events
//
// listen() registers a handler for a named event and returns an unlisten fn,
// exactly matching the Tauri API.

type Handler<T> = (event: { payload: T }) => void;

const handlers = new Map<string, Handler<any>[]>();
let   ws: WebSocket | null = null;
let   reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/api/events`);

  ws.onopen = () => {
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  };

  ws.onmessage = (e: MessageEvent) => {
    let msg: any;
    try { msg = JSON.parse(e.data as string); } catch { return; }

    // Dispatch to handlers by event name.
    // Server sends either { event, payload } (viz events) or progress { tool, pct }.
    const key = msg.event ?? msg.tool;
    if (!key) return;
    const list = handlers.get(key);
    if (list) {
      const wrappedPayload = msg.payload !== undefined ? msg.payload : msg;
      list.forEach(h => h({ payload: wrappedPayload }));
    }
  };

  ws.onclose = () => {
    ws = null;
    // Reconnect after 2 s — handles server restart or temporary drop
    reconnectTimer = setTimeout(connect, 2000);
  };

  ws.onerror = () => {
    ws?.close();
  };
}

export async function listen<T>(
  event: string,
  handler: Handler<T>,
): Promise<() => void> {
  connect();
  if (!handlers.has(event)) handlers.set(event, []);
  handlers.get(event)!.push(handler);

  // Return an unlisten function, matching the Tauri API
  return () => {
    const arr = handlers.get(event);
    if (arr) handlers.set(event, arr.filter(h => h !== handler));
  };
}
