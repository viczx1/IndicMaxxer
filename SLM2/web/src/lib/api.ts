export const API = process.env.NEXT_PUBLIC_API ?? "http://127.0.0.1:8000";   // not "localhost": Windows tries IPv6 first (+2 s per request)

export type Trace = {
  style: string; mode: string; tools: string[]; sources: [string, string][];
  think?: string; prompt_tokens?: number; new_tokens?: number; seconds?: number; tok_per_s?: number;
  tax?: Record<string, number>; doc_tax?: Record<string, number>; vote?: string; compress?: { before: number; after: number }; processed?: Record<string, number>;
};
export type Pass = { think: string; answer: string };
export type ChatEvent =
  | { type: "status"; text: string }
  | { type: "meta"; style: string; mode: string; tools: string[]; sources: [string, string][] }
  | { type: "think"; text: string }
  | { type: "token"; text: string }
  | { type: "replace"; text: string }
  | { type: "passes"; passes: Pass[]; vote: string }
  | { type: "done"; answer: string; trace: Trace; file: string | null; docs?: string[] }
  | { type: "error"; message: string };

export type Msg = { role: "user" | "assistant"; content: string };

export async function health() {
  const r = await fetch(`${API}/api/health`, { cache: "no-store" });
  return r.json() as Promise<{ status: string; error: string | null; gpu: string | null; quant: string; tax_refs: string[] }>;
}

/** POST /api/chat and call onEvent for every server-sent event. */
export async function streamChat(
  opts: { message: string; history: Msg[]; mode: string; k: number; forceSearch: boolean; files?: File[]; compress?: boolean; docs?: string[]; style?: string; voice?: boolean },
  onEvent: (e: ChatEvent) => void, signal?: AbortSignal,
) {
  const fd = new FormData();
  fd.append("message", opts.message);
  fd.append("history", JSON.stringify(opts.history.slice(-6)));
  fd.append("mode", opts.mode);
  fd.append("k", String(opts.k));
  fd.append("force_search", String(opts.forceSearch));
  fd.append("compress", String(opts.compress ?? true));
  fd.append("docs", JSON.stringify(opts.docs ?? []));
  if (opts.style) fd.append("style", opts.style);
  if (opts.voice) fd.append("voice", "true");
  for (const f of opts.files ?? []) fd.append("files", f);
  const r = await fetch(`${API}/api/chat`, { method: "POST", body: fd, signal });
  if (!r.ok || !r.body) throw new Error(`backend ${r.status}`);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice(6)));
    }
  }
}

export async function tts(text: string, style?: string, voice = "charvi"): Promise<Blob | null> {
  const r = await fetch(`${API}/api/tts`, { method: "POST", headers: { "Content-Type": "application/json" },
                                           body: JSON.stringify({ text, style, voice }) });
  return r.status === 200 ? r.blob() : null;
}

/** Local speech-to-text (Whisper on the laptop). `wav` is 16 kHz mono PCM16. */
export async function stt(wav: Blob, lang: string, verify = true): Promise<{ text: string; seconds?: number; rejected?: boolean; similarity?: number }> {
  const fd = new FormData();
  fd.append("audio", wav, "speech.wav");
  fd.append("lang", lang);
  fd.append("verify", String(verify));
  const r = await fetch(`${API}/api/stt`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`stt ${r.status}`);
  return r.json();
}

export function sttWarm(lang: string) {
  const fd = new FormData(); fd.append("lang", lang);
  return fetch(`${API}/api/stt/warm`, { method: "POST", body: fd }).catch(() => null);
}

/** Float32 samples at `rate` -> 16 kHz mono 16-bit WAV. */
export function toWav16k(chunks: Float32Array[], rate: number): Blob {
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const all = new Float32Array(n); let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
  const ratio = rate / 16000, len = Math.floor(n / ratio);
  const pcm = new Int16Array(len);
  for (let i = 0; i < len; i++) {
    const a = Math.floor(i * ratio), b = Math.min(n - 1, Math.floor((i + 1) * ratio));
    let s = 0; for (let j = a; j <= b; j++) s += all[j]; s /= (b - a + 1);       // box filter = cheap anti-alias
    pcm[i] = Math.max(-1, Math.min(1, s)) * 0x7fff;
  }
  const buf = new ArrayBuffer(44 + pcm.byteLength), v = new DataView(buf);
  const w = (p: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(p + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + pcm.byteLength, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, 16000, true);
  v.setUint32(28, 32000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, pcm.byteLength, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], { type: "audio/wav" });
}

/** Only-my-voice: enrol a sample of the user's voice, check status, or reset. */
export async function enrollVoice(wav: Blob, reset = false) {
  const fd = new FormData(); fd.append("audio", wav, "me.wav"); fd.append("reset", String(reset));
  const r = await fetch(`${API}/api/voiceprint`, { method: "POST", body: fd });
  return r.json() as Promise<{ ok: boolean; seconds?: number; error?: string }>;
}
export const voiceprintStatus = () => fetch(`${API}/api/voiceprint`).then((r) => r.json() as Promise<{ enrolled: boolean }>);
export const resetVoiceprint = () => fetch(`${API}/api/voiceprint`, { method: "DELETE" }).then((r) => r.json());
