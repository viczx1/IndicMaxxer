"use client";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Paperclip, Send, Square, Search, Minimize2, Brain, Zap, FileDown, X, ChevronDown, Trash2 } from "lucide-react";
import Aurora from "@/components/Aurora";
import Nav from "@/components/Nav";
import { API, ChatEvent, Msg, Pass, Trace, streamChat } from "@/lib/api";

type Bubble = { role: "user" | "assistant"; content: string; think?: string; status?: string; tools?: string[];
                sources?: [string, string][]; passes?: Pass[]; vote?: string; trace?: Trace; file?: string | null;
                error?: string; attachments?: string[] };

const SUGGEST = ["naale client meeting ide, enu prepare maadli?", "ಮೈಸೂರು ಅರಮನೆ ಬಗ್ಗೆ ಹುಡುಕಿ ಹೇಳು",
                 "kal interview hai, kuch tips do", "who are you?"];
const ease = [0.16, 1, 0.3, 1] as const;

function Think({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(false);
  if (!text.trim()) return null;
  return (
    <div className="mb-2">
      <button onClick={() => setOpen(!open)} className="kicker flex items-center gap-1.5 hover:text-lime">
        <Brain size={12} /> {live ? "thinking…" : "thought process"} <ChevronDown size={12} className={`transition ${open ? "rotate-180" : ""}`} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden">
            <p className="mt-2 whitespace-pre-wrap border-l-2 border-lime/30 pl-3 text-sm text-[var(--muted)]">{text}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function TaxPanel({ trace }: { trace?: Trace }) {
  if (!trace?.tax) return <p className="text-sm text-[var(--muted)]">Send a message to see what it costs in tokens.</p>;
  const cz = trace.compress;
  const base = trace.tax["IndicMaxxer-2B"] || 1;
  const max = Math.max(...Object.values(trace.tax));
  return (
    <div className="space-y-3">
      {Object.entries(trace.tax).map(([k, v]) => (
        <div key={k}>
          <div className="flex justify-between text-xs"><span className={k === "IndicMaxxer-2B" ? "text-lime" : "text-[var(--muted)]"}>{k}</span>
            <span className="font-mono">{v}{k !== "IndicMaxxer-2B" && <span className="text-[var(--muted)]"> ({(v / base).toFixed(1)}×)</span>}</span></div>
          <div className="mt-1 h-1.5 rounded-full bg-white/5">
            <motion.div layout initial={{ width: 0 }} animate={{ width: `${(v / max) * 100}%` }} transition={{ duration: 0.9, ease }}
                        className={`h-full rounded-full ${k === "IndicMaxxer-2B" ? "bg-lime shadow-[0_0_10px_#8be04e]" : "bg-rose-300/70"}`} />
          </div>
        </div>
      ))}
      {trace.processed && (() => {
        const ours = trace.processed["IndicMaxxer-2B"], gpt = trace.processed["Sarvam-1"], win = ours <= gpt;
        return (
          <div className="glass-dark p-3 text-sm">
            <div className="flex justify-between"><span className="text-[var(--muted)]">processed this turn</span>
              <b className={win ? "text-lime" : "text-amber-200"}>{win ? `${(gpt / Math.max(ours, 1)).toFixed(2)}× fewer` : `${(ours / Math.max(gpt, 1)).toFixed(2)}× more`}</b></div>
            <div className="mt-1 text-xs text-[var(--muted)]">IndicMaxxer {ours} · stock Sarvam-1 {gpt} tokens, counting history, search results, documents and the reply (what you actually pay for)</div>
          </div>
        );
      })()}
      {cz && cz.before > 0 && (
        <div className="glass-dark p-3 text-sm">
          <div className="flex justify-between"><span className="text-[var(--muted)]">compressor</span>
            <b className="text-lime">−{Math.round((100 * (cz.before - cz.after)) / cz.before)}%</b></div>
          <div className="mt-1 text-xs text-[var(--muted)]">context {cz.before} → {cz.after} tokens before the model reads it</div>
        </div>
      )}
      <div className="grid grid-cols-3 gap-2 pt-2 text-center">
        {[["sec", trace.seconds], ["tok/s", trace.tok_per_s], ["style", trace.style]].map(([l, v]) => (
          <div key={l as string} className="rounded-lg bg-white/5 py-2"><div className="font-mono text-sm">{String(v ?? "–")}</div><div className="kicker !text-[9px] !tracking-wider">{l}</div></div>
        ))}
      </div>
    </div>
  );
}

export default function Chat() {
  const [msgs, setMsgs] = useState<Bubble[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<"instruct" | "reasoning">("instruct");
  const [k, setK] = useState(3);
  const [force, setForce] = useState(false);
  const [squeeze, setSqueeze] = useState(true);
  const [docs, setDocs] = useState<string[]>([]);          // documents read in this chat: follow-ups can ask about them
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const fileIn = useRef<HTMLInputElement>(null);
  const last = [...msgs].reverse().find((m) => m.trace)?.trace;

  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" }); }, [msgs]);

  const patch = (fn: (b: Bubble) => Bubble) => setMsgs((m) => [...m.slice(0, -1), fn(m[m.length - 1])]);

  async function send(text = input) {
    if (busy || (!text.trim() && !files.length)) return;
    const history: Msg[] = msgs.filter((m) => !m.error).map((m) => ({ role: m.role, content: m.content }));
    setMsgs((m) => [...m, { role: "user", content: text, attachments: files.map((f) => f.name) },
                        { role: "assistant", content: "", think: "", status: "connecting" }]);
    setInput(""); setBusy(true);
    const sent = files; setFiles([]);
    abort.current = new AbortController();
    try {
      await streamChat({ message: text, history, mode, k, forceSearch: force, files: sent, compress: squeeze, docs }, (e: ChatEvent) => {
        if (e.type === "status") patch((b) => ({ ...b, status: e.text }));
        else if (e.type === "meta") patch((b) => ({ ...b, tools: e.tools, sources: e.sources }));
        else if (e.type === "think") patch((b) => ({ ...b, think: (b.think ?? "") + e.text }));
        else if (e.type === "token") patch((b) => ({ ...b, content: b.content + e.text, status: undefined }));
        else if (e.type === "replace") patch((b) => ({ ...b, content: e.text }));
        else if (e.type === "passes") patch((b) => ({ ...b, passes: e.passes, vote: e.vote, think: "" }));
        else if (e.type === "done" && e.docs?.length) setDocs((d) => [...d, ...e.docs!].slice(-3));
        if (e.type === "done") patch((b) => ({ ...b, content: e.answer || b.content, trace: e.trace, tools: e.trace.tools, file: e.file, status: undefined }));
        else if (e.type === "error") patch((b) => ({ ...b, error: e.message, status: undefined }));
      }, abort.current.signal);
    } catch (err) {
      if ((err as Error).name !== "AbortError")
        patch((b) => ({ ...b, error: `Can't reach the backend at ${API}. Is server.py running?`, status: undefined }));
    } finally { setBusy(false); }
  }

  return (
    <main className="relative h-screen overflow-hidden">
      <Aurora calm />
      <Nav />
      <div className="relative z-10 mx-auto grid h-full max-w-7xl grid-cols-1 gap-4 px-4 pb-4 pt-20 lg:grid-cols-[1fr_300px]">
        {/* chat window */}
        <motion.section initial={{ opacity: 0, y: 24, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.8, ease }}
                        className="glass flex min-h-0 flex-col">
          <div ref={scroller} data-lenis-prevent className="thin-scroll flex-1 space-y-5 overflow-y-auto px-4 py-6 sm:px-8">
            {msgs.length === 0 && (
              <div className="grid h-full place-items-center text-center">
                <div>
                  <h2 className="soft-glow text-4xl font-semibold">Namaskara.</h2>
                  <p className="mt-2 text-[var(--muted)]">Ask in Kannada, Hindi, English, Kanglish or Hinglish.</p>
                  <div className="mt-6 flex flex-wrap justify-center gap-2">
                    {SUGGEST.map((s) => <button key={s} onClick={() => send(s)} className="pill px-4 py-2 text-sm">{s}</button>)}
                  </div>
                </div>
              </div>
            )}
            <AnimatePresence initial={false}>
              {msgs.map((m, i) => (
                <motion.div key={i} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease }}
                            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div className={`max-w-[85%] rounded-2xl px-4 py-3 ${m.role === "user"
                    ? "bg-gradient-to-b from-[#7fd85a]/45 to-[#2f9a35]/35 ring-1 ring-lime/40 shadow-[inset_0_1px_0_rgba(255,255,255,.25)]"
                    : "bg-white/[0.04] ring-1 ring-white/10"}`}>
                    {m.attachments?.length ? <div className="kicker mb-1">📎 {m.attachments.join(", ")}</div> : null}
                    {m.role === "assistant" && m.tools && m.tools.length > 0 && (
                      <div className="mb-2 flex flex-wrap gap-1.5">
                        {m.tools.map((t, j) => <span key={j} className="rounded-full bg-sky/10 px-2 py-0.5 font-mono text-[10px] text-sky ring-1 ring-sky/30">{t.split(":")[0]}</span>)}
                      </div>
                    )}
                    {m.role === "assistant" && <Think text={m.think ?? ""} live={!!m.status} />}
                    {m.passes && m.passes.length > 1 && (
                      <details className="mb-2 text-sm"><summary className="kicker cursor-pointer hover:text-lime">{m.passes.length} native passes · {m.vote}</summary>
                        <div className="mt-2 space-y-2">{m.passes.map((p, j) => (
                          <div key={j} className="rounded-lg bg-black/15 p-2 text-xs text-[var(--muted)]"><b className="text-lime">pass {j + 1}</b> {p.think.slice(0, 400)}<div className="mt-1 text-[var(--fg)]">→ {p.answer.slice(0, 200)}</div></div>
                        ))}</div>
                      </details>
                    )}
                    {m.status && !m.content && (
                      <div className="flex items-center gap-2 text-sm text-[var(--muted)]">
                        <span className="flex gap-1">{[0, 1, 2].map((d) => <motion.span key={d} className="h-1.5 w-1.5 rounded-full bg-lime"
                          animate={{ opacity: [0.2, 1, 0.2], y: [0, -3, 0] }} transition={{ repeat: Infinity, duration: 1, delay: d * 0.15 }} />)}</span>
                        {m.status}
                      </div>
                    )}
                    <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
                    {m.error && <div className="mt-1 text-sm text-rose-300">{m.error}</div>}
                    {m.file && <a href={`${API}${m.file}`} className="pill mt-3 inline-flex items-center gap-2 px-3 py-1.5 text-sm"><FileDown size={14} /> download document</a>}
                    {m.sources && m.sources.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">{m.sources.map(([t, u], j) => (
                        <a key={j} href={u} target="_blank" rel="noreferrer" className="rounded-full bg-white/5 px-2 py-0.5 text-[11px] text-[var(--muted)] ring-1 ring-white/10 hover:text-lime">{t.slice(0, 40)}</a>
                      ))}</div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
          {/* composer */}
          <div className="border-t border-white/10 p-3">
            {docs.length > 0 && files.length === 0 && (
              <div className="mb-2 flex items-center gap-2 text-xs text-[var(--muted)]">
                <span className="rounded-full bg-white/10 px-3 py-1 ring-1 ring-white/20">📄 {docs.length} document{docs.length > 1 ? "s" : ""} in this chat: ask anything about {docs.length > 1 ? "them" : "it"}</span>
                <button onClick={() => setDocs([])} className="hover:text-white">forget</button>
              </div>
            )}
            {files.length > 0 && <div className="mb-2 flex flex-wrap gap-2">{files.map((f, i) => (
              <span key={i} className="flex items-center gap-1 rounded-full bg-white/5 px-3 py-1 text-xs ring-1 ring-white/10">{f.name}
                <button onClick={() => setFiles(files.filter((_, j) => j !== i))}><X size={12} /></button></span>))}</div>}
            <div className="flex items-end gap-2">
              <input ref={fileIn} type="file" multiple hidden accept=".pdf,.docx,.txt,.md,image/*"
                     onChange={(e) => { setFiles([...files, ...Array.from(e.target.files ?? [])]); e.target.value = ""; }} />
              <button onClick={() => fileIn.current?.click()} className="pill !p-0 justify-center grid h-11 w-11 shrink-0 place-items-center" title="attach PDF / DOCX / image"><Paperclip size={18} /></button>
              <textarea value={input} onChange={(e) => setInput(e.target.value)} rows={1} placeholder="ಏನು ಬೇಕು? · kya chahiye? · ask anything…"
                        onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                        className="max-h-40 min-h-11 flex-1 resize-none rounded-2xl bg-black/15 px-4 py-2.5 outline-none ring-1 ring-white/10 placeholder:text-[var(--muted)] focus:ring-lime/50" />
              {busy
                ? <button onClick={() => abort.current?.abort()} className="pill !p-0 justify-center grid h-11 w-11 shrink-0 place-items-center" title="stop"><Square size={16} /></button>
                : <button onClick={() => send()} className="pill pill-green !p-0 justify-center grid h-11 w-11 shrink-0 place-items-center" title="send"><Send size={17} /></button>}
            </div>
          </div>
        </motion.section>

        {/* side panel */}
        <motion.aside initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.8, delay: 0.1, ease }}
                      className="hidden min-h-0 flex-col gap-4 lg:flex">
          <div className="glass p-4">
            <div className="kicker mb-3">mode</div>
            <div className="relative grid grid-cols-2 rounded-full bg-black/15 p-1 ring-1 ring-white/10">
              {(["instruct", "reasoning"] as const).map((m) => (
                <button key={m} onClick={() => setMode(m)} className="relative z-10 flex items-center justify-center gap-1.5 rounded-full py-2 text-sm">
                  {mode === m && <motion.span layoutId="modepill" className="absolute inset-0 -z-10 rounded-full bg-lime/20 ring-1 ring-lime/50" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
                  {m === "instruct" ? <Zap size={14} /> : <Brain size={14} />} /{m}
                </button>
              ))}
            </div>
            <p className="mt-3 text-xs text-[var(--muted)]">{mode === "instruct" ? "Direct answer. The router calls tools when they're needed." : `Thinks natively ${k}× and votes on the answer.`}</p>
            <AnimatePresence>{mode === "reasoning" && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                <div className="mt-4 flex items-center justify-between text-xs"><span className="kicker">passes (k)</span><span className="font-mono">{k}</span></div>
                <input type="range" min={1} max={5} value={k} onChange={(e) => setK(+e.target.value)} className="w-full" />
              </motion.div>)}</AnimatePresence>
            <label className="mt-4 flex cursor-pointer items-center justify-between text-sm">
              <span className="flex items-center gap-2"><Search size={14} /> force web search</span>
              <button onClick={() => setForce(!force)} className={`relative h-5 w-9 rounded-full transition ${force ? "bg-lime/60" : "bg-white/10"}`}>
                <motion.span layout className="absolute top-0.5 h-4 w-4 rounded-full bg-white shadow" style={{ left: force ? 18 : 2 }} transition={{ type: "spring", stiffness: 500, damping: 30 }} />
              </button>
            </label>
            <label className="mt-4 flex cursor-pointer items-center justify-between text-sm">
              <span className="flex items-center gap-2"><Minimize2 size={14} /> token compressor</span>
              <button onClick={() => setSqueeze(!squeeze)} className={`relative h-5 w-9 rounded-full transition ${squeeze ? "bg-lime/60" : "bg-white/10"}`}>
                <motion.span layout className="absolute top-0.5 h-4 w-4 rounded-full bg-white shadow" style={{ left: squeeze ? 18 : 2 }} transition={{ type: "spring", stiffness: 500, damping: 30 }} />
              </button>
            </label>
          </div>
          <div data-lenis-prevent className="glass flex-1 overflow-y-auto p-4 thin-scroll">
            <div className="kicker mb-3">token tax · last turn</div>
            <TaxPanel trace={last} />
            {last?.tools && last.tools.length > 0 && (<><div className="kicker mb-2 mt-5">tools</div>
              <ul className="space-y-1 font-mono text-[11px] text-[var(--muted)]">{last.tools.map((t, i) => <li key={i}>› {t}</li>)}</ul></>)}
          </div>
          <button onClick={() => { if (!busy) { setMsgs([]); setDocs([]); } }} className="pill flex items-center justify-center gap-2 py-2 text-sm"><Trash2 size={14} /> new chat</button>
        </motion.aside>
      </div>
    </main>
  );
}
