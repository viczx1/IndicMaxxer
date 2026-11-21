"use client";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { AudioLines, Keyboard, Mic, Square, Globe, ChevronDown, User, Repeat, Send } from "lucide-react";
import Aurora from "@/components/Aurora";
import Nav from "@/components/Nav";
import Orb, { OrbState, Wave } from "@/components/Orb";
import { Msg, streamChat, stt, sttWarm, toWav16k, tts } from "@/lib/api";

// Spoken Kanglish is spoken Kannada (same for Hinglish/Hindi): the Kannada / Hindi recogniser handles it.
const LANGS = [
  { id: "kn", label: "ಕನ್ನಡ · Kanglish", name: "Kannada" },
  { id: "hi", label: "हिंदी · Hinglish", name: "Hindi" },
  { id: "en", label: "English", name: "English" },
];
const VOICES = [{ id: "charvi", label: "Charvi" }, { id: "satvik", label: "Satvik" }];
const STATUS: Record<OrbState, string> = { idle: "Tap the mic and speak", listening: "Listening…", thinking: "Thinking…", speaking: "Speaking… (just talk to interrupt)" };
const ease = [0.16, 1, 0.3, 1] as const;
const END_SILENCE_MS = 1600;   // you finished talking after this much silence (short pauses mid-sentence are fine)
const START_MS = 120;          // this much voice starts a recording
const BARGE_MS = 260;          // this much of your voice while it speaks = interruption
const PREROLL_S = 0.45;        // keep the start of your first word
const MAX_UTTER_S = 25;
type Turn = { you: string; bot: string; cut?: boolean };

export default function Voice() {
  const [state, setState] = useState<OrbState>("idle");
  const [phase, setPhase] = useState<"" | "hearing" | "transcribing">("");
  const [lang, setLang] = useState(LANGS[0]);
  const [voice, setVoice] = useState(VOICES[0]);
  const [langOpen, setLangOpen] = useState(false);
  const [textMode, setTextMode] = useState(false);
  const [typed, setTyped] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [err, setErr] = useState("");
  const [loop, setLoop] = useState(true);
  const level = useRef(0);
  const actx = useRef<AudioContext | null>(null);
  const mic = useRef<{ stream: MediaStream } | null>(null);
  const rec = useRef({ on: false, chunks: [] as Float32Array[], pre: [] as Float32Array[], preLen: 0, voiced: 0, silent: 0, total: 0 });
  const history = useRef<Msg[]>([]);
  const cancel = useRef<() => void>(() => {});
  const stateRef = useRef(state); stateRef.current = state;
  const loopRef = useRef(loop); loopRef.current = loop;
  const textRef = useRef(textMode); textRef.current = textMode;
  const langRef = useRef(lang); langRef.current = lang;
  const voiceRef = useRef(voice); voiceRef.current = voice;
  const feed = useRef<HTMLDivElement>(null);

  useEffect(() => { feed.current?.scrollTo({ top: feed.current.scrollHeight, behavior: "smooth" }); }, [turns, phase]);
  useEffect(() => { sttWarm(lang.id); }, [lang]);           // load the recogniser before you speak

  const ctx = () => (actx.current ??= new AudioContext());

  /** Mic open for the whole session (echo-cancelled). Our own voice detector decides when you start and stop. */
  async function ensureMic() {
    if (mic.current) return;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    mic.current = { stream };
    const ac = ctx(), src = ac.createMediaStreamSource(stream);
    const proc = ac.createScriptProcessor(2048, 1, 1);
    const mute = ac.createGain(); mute.gain.value = 0;
    src.connect(proc); proc.connect(mute); mute.connect(ac.destination);
    let floor = 0.008, smooth = 0;
    proc.onaudioprocess = (e) => {
      const x = new Float32Array(e.inputBuffer.getChannelData(0));
      const ms = (x.length / ac.sampleRate) * 1000;
      let s = 0; for (const q of x) s += q * q; const raw = Math.sqrt(s / x.length);
      smooth = raw > smooth ? raw : smooth * 0.85 + raw * 0.15;           // fast attack, slow release
      const v = smooth;
      const st = stateRef.current, r = rec.current;
      const thr = Math.max(st === "speaking" ? 0.05 : 0.018, floor * (st === "speaking" ? 4 : 3));
      if (!r.on && st !== "speaking") floor = floor * 0.97 + Math.min(v, 0.04) * 0.03;     // learn the room's noise
      if (st === "listening" || r.on) level.current = Math.min(1, v * 6);
      // keep a short pre-roll so the first syllable isn't cut
      r.pre.push(x); r.preLen += x.length;
      while (r.preLen - r.pre[0].length > PREROLL_S * ac.sampleRate) r.preLen -= r.pre.shift()!.length;
      if (st === "speaking") {                              // barge-in
        r.voiced = v > thr ? r.voiced + ms : Math.max(0, r.voiced - ms / 2);
        if (r.voiced > BARGE_MS) { r.voiced = 0; interrupt(); }
        return;
      }
      if (st !== "listening") return;
      if (!r.on) {
        r.voiced = v > thr ? r.voiced + ms : 0;
        if (r.voiced > START_MS) { r.on = true; r.chunks = [...r.pre]; r.silent = 0; r.total = 0; setPhase("hearing"); }
        return;
      }
      r.chunks.push(x); r.total += ms;
      const keep = Math.max(0.008, floor * 1.8);                         // lower bar to *continue* than to start
      r.silent = v > keep ? 0 : r.silent + ms;
      if (r.silent > END_SILENCE_MS || r.total > MAX_UTTER_S * 1000) endUtterance();
    };
  }

  async function listen() {
    setErr("");
    try { await ensureMic(); } catch { setErr("Microphone permission was denied."); return; }
    ctx().resume();
    Object.assign(rec.current, { on: false, chunks: [], voiced: 0, silent: 0, total: 0 });
    setPhase(""); setState("listening");
  }

  async function endUtterance() {
    const r = rec.current;
    if (!r.on) return;
    r.on = false;
    const wav = toWav16k(r.chunks, ctx().sampleRate);
    r.chunks = [];
    setPhase("transcribing"); setState("thinking");
    try {
      const { text } = await stt(wav, langRef.current.id);
      setPhase("");
      if (text.trim()) think(text.trim());
      else if (loopRef.current) listen(); else setState("idle");
    } catch {
      setPhase(""); setState("idle");
      setErr("Speech recognition failed. Is the backend running?");
    }
  }

  function interrupt() {
    if (stateRef.current !== "speaking" && stateRef.current !== "thinking") return;
    cancel.current();
    setTurns((t) => t.length ? [...t.slice(0, -1), { ...t[t.length - 1], cut: true }] : t);
    listen().then(() => {                                   // you are already talking: start recording right away
      const r = rec.current; r.on = true; r.chunks = [...r.pre]; r.silent = 0; r.total = 0; setPhase("hearing");
    });
  }

  async function think(text: string) {
    rec.current.on = false;
    setTurns((t) => [...t, { you: text, bot: "" }]);
    setState("thinking"); setPhase("");
    let stopped = false, style: string = langRef.current.id, said = "", spoken = 0, doneStreaming = false;
    const ac = new AbortController();
    const queue: Promise<Blob | null>[] = [];
    let playing: AudioBufferSourceNode | null = null;
    cancel.current = () => { stopped = true; ac.abort(); try { playing?.stop(); } catch {} level.current = 0; };
    const setBot = (s: string) => setTurns((t) => [...t.slice(0, -1), { ...t[t.length - 1], bot: s }]);
    const speak = (s: string) => queue.push(tts(s, style, voiceRef.current.id).catch(() => null));
    const pushSentences = (flush: boolean) => {
      let rest = said.slice(spoken), end = 0;
      if (queue.length === 0) {                            // first words out fast: the first piece may end at a comma
        const first = /^[^.!?।\n,]{12,}[.!?।\n,]/.exec(rest);
        if (first) { speak(first[0].trim()); spoken += first[0].length; rest = rest.slice(first[0].length); }
      }
      const re = /[^.!?।\n]+[.!?।\n]+/g; let m: RegExpExecArray | null;
      while ((m = re.exec(rest))) { const s = m[0].trim(); if (s.length > 1) speak(s); end = m.index + m[0].length; }
      if (flush && rest.slice(end).trim()) { speak(rest.slice(end).trim()); end = rest.length; }
      spoken += end;
    };
    const player = (async () => {
      for (let i = 0; ; i++) {
        while (i >= queue.length) { if (stopped || doneStreaming) return; await new Promise((r) => setTimeout(r, 40)); }
        const blob = await queue[i]; if (stopped) return; if (!blob) continue;
        const buf = await ctx().decodeAudioData(await blob.arrayBuffer());
        if (stopped) return;
        const src = ctx().createBufferSource(); src.buffer = buf; playing = src;
        const an = ctx().createAnalyser(); an.fftSize = 512; src.connect(an); an.connect(ctx().destination);
        const b = new Uint8Array(new ArrayBuffer(an.fftSize)); let on = true;
        const meter = () => {
          if (!on) return; an.getByteTimeDomainData(b); let s = 0; for (const v of b) s += ((v - 128) / 128) ** 2;
          level.current = Math.min(1, Math.sqrt(s / b.length) * 4); requestAnimationFrame(meter);
        };
        setState("speaking"); meter();
        await new Promise<void>((res) => { src.onended = () => res(); src.start(); });
        on = false; level.current = 0;
      }
    })();
    try {
      await streamChat({ message: text, history: history.current, mode: "instruct", k: 1, forceSearch: false }, (e) => {
        if (e.type === "meta") style = e.style;
        else if (e.type === "token") { said += e.text; setBot(said); pushSentences(false); }
        else if (e.type === "replace") { said = e.text; spoken = 0; setBot(said); }
        else if (e.type === "done") { said = e.answer || said; setBot(said); }
        else if (e.type === "error") setErr(e.message);
      }, ac.signal);
      pushSentences(true);
    } catch (e) { if ((e as Error).name !== "AbortError") setErr("Can't reach the local model. Is the backend running?"); }
    doneStreaming = true;
    await player;
    history.current = [...history.current, { role: "user" as const, content: text }, { role: "assistant" as const, content: said }].slice(-6);
    if (!stopped) { if (loopRef.current && !textRef.current) listen(); else setState("idle"); }
  }

  function micButton() {
    ctx().resume();
    if (state === "idle") listen();
    else if (state === "listening") { if (rec.current.on) endUtterance(); else { setState("idle"); setPhase(""); } }
    else interrupt();
  }
  function stop() { cancel.current(); rec.current.on = false; setState("idle"); setPhase(""); level.current = 0; setLoop(false); }
  function sendTyped() {                                     // works in any state: stops listening/speaking first
    const t = typed.trim(); if (!t) return;
    cancel.current(); rec.current.on = false; ctx().resume(); setTyped(""); think(t);
  }

  useEffect(() => () => { cancel.current(); mic.current?.stream.getTracks().forEach((t) => t.stop()); actx.current?.close(); }, []);

  const status = phase === "hearing" ? "Hearing you…" : phase === "transcribing" ? "Understanding…" : STATUS[state];
  return (
    <main className="relative min-h-screen">
      <Aurora calm />
      <Nav />
      <section className="relative z-10 mx-auto flex min-h-screen max-w-6xl items-center px-4 pb-8 pt-24">
        <motion.div initial={{ opacity: 0, y: 30, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.9, ease }}
                    className="glass w-full p-5 sm:p-8">
          <div className="grid items-center gap-6 lg:grid-cols-[1.15fr_1fr]">
            <div className="flex flex-col items-center">
              <div onClick={micButton} className="cursor-pointer"><Orb state={state} level={level} /></div>
              <AnimatePresence mode="wait">
                <motion.div key={status} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.25 }}
                            className="display soft-glow -mt-6 text-center text-2xl font-light sm:text-3xl">{status}</motion.div>
              </AnimatePresence>
              <Wave level={level} active={state === "listening" || state === "speaking"} />
            </div>
            <div className="glass-inset flex h-[420px] flex-col p-3">
              <div ref={feed} data-lenis-prevent className="thin-scroll flex-1 space-y-3 overflow-y-auto p-1">
                {turns.length === 0 && (
                  <div className="grid h-full place-items-center px-6 text-center text-[var(--muted)]">
                    <div>
                      <p className="text-lg text-white">Talk to IndicMaxxer in {lang.name}.</p>
                      <p className="mt-2 text-sm">Speak naturally, it waits until you finish. Start talking while it speaks to interrupt it. Everything runs on your laptop.</p>
                    </div>
                  </div>
                )}
                {turns.map((t, i) => (
                  <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease }} className="space-y-3">
                    <div className="glass-dark flex gap-3 p-3">
                      <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-white/10 ring-1 ring-white/25"><User size={18} /></span>
                      <div><div className="text-sm text-[var(--muted)]">You said:</div><div className="text-[17px]">“{t.you}”</div></div>
                    </div>
                    <div className="flex gap-3 p-3">
                      <span className="bead mt-0.5 h-10 w-10 shrink-0" />
                      <div><div className="text-sm font-medium text-lime">IndicMaxxer:{t.cut && <span className="ml-2 text-xs text-amber-200">(interrupted)</span>}</div>
                        <div className="whitespace-pre-wrap text-[17px] leading-relaxed">{t.bot || <span className="text-[var(--muted)]">…</span>}</div></div>
                    </div>
                  </motion.div>
                ))}
                {phase && <div className="glass-dark p-3 text-[var(--muted)]">{phase === "hearing" ? "🎙 hearing you…" : "✍ transcribing on your laptop…"}</div>}
              </div>
              {err && <p className="px-2 pt-2 text-sm text-amber-200">{err}</p>}
            </div>
          </div>

          <div className="glass-inset mt-6 flex flex-wrap items-center justify-center gap-3 p-3 sm:justify-between">
            <div className="flex gap-2">
              <button onClick={() => setTextMode(false)} className={`pill px-5 py-2.5 ${!textMode ? "pill-on" : ""}`}><AudioLines size={18} /> Voice mode</button>
              <button onClick={() => { setTextMode(true); if (state === "listening") { rec.current.on = false; setState("idle"); setPhase(""); } }}
                      className={`pill px-5 py-2.5 ${textMode ? "pill-on" : ""}`}><Keyboard size={18} /> Text mode</button>
            </div>
            {textMode ? (
              <div className="flex min-w-[260px] flex-1 items-center gap-2 sm:max-w-md">
                <input value={typed} onChange={(e) => setTyped(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendTyped()}
                       placeholder={`Type in ${lang.name}…`} className="glass-dark h-12 flex-1 !rounded-full px-5 outline-none placeholder:text-[var(--faint)] focus:ring-2 focus:ring-sky/60" />
                <button onClick={sendTyped} className="ring-btn grid h-12 w-12 place-items-center" aria-label="send"><Send size={18} className="relative" /></button>
              </div>
            ) : (
              <motion.button whileTap={{ scale: 0.94 }} onClick={micButton} aria-label="microphone" className="ring-btn grid h-[72px] w-[72px] place-items-center">
                {state === "listening"
                  ? <motion.span animate={{ scale: [1, 1.15, 1] }} transition={{ repeat: Infinity, duration: 1 }}><Mic size={30} className="relative text-lime drop-shadow" /></motion.span>
                  : state === "idle" ? <Mic size={30} className="relative text-lime drop-shadow" /> : <Square size={24} className="relative text-white" />}
              </motion.button>
            )}
            <div className="flex flex-wrap gap-2">
              <div className="relative">
                <button onClick={() => setLangOpen(!langOpen)} className="pill px-5 py-2.5"><Globe size={18} /> {lang.label} <ChevronDown size={16} className={`transition ${langOpen ? "rotate-180" : ""}`} /></button>
                <AnimatePresence>
                  {langOpen && (
                    <motion.ul initial={{ opacity: 0, y: 8, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 8, scale: 0.97 }}
                               transition={{ duration: 0.18 }} className="glass absolute bottom-14 left-0 z-20 w-56 !rounded-2xl p-1.5">
                      {LANGS.map((l) => (
                        <li key={l.id}><button onClick={() => { setLang(l); setLangOpen(false); }}
                          className={`w-full rounded-xl px-3 py-2 text-left hover:bg-white/20 ${l.id === lang.id ? "bg-white/20" : ""}`}>{l.label}</button></li>
                      ))}
                      <li className="kicker px-3 pb-1 pt-2">voice</li>
                      {VOICES.map((v) => (
                        <li key={v.id}><button onClick={() => { setVoice(v); setLangOpen(false); }}
                          className={`w-full rounded-xl px-3 py-2 text-left hover:bg-white/20 ${v.id === voice.id ? "bg-white/20" : ""}`}>{v.label}</button></li>
                      ))}
                    </motion.ul>
                  )}
                </AnimatePresence>
              </div>
              <button onClick={() => setLoop(!loop)} title="Conversation mode: keeps listening after each reply" className={`pill px-4 py-2.5 ${loop ? "pill-on" : ""}`}><Repeat size={17} /> {loop ? "Conversation" : "One turn"}</button>
              <button onClick={stop} className="pill px-5 py-2.5"><Square size={16} className="fill-white" /> Stop</button>
            </div>
          </div>
          <p className="mt-4 text-center text-xs text-[var(--faint)]">Runs fully on your laptop: Whisper (Kannada / Hindi / English) hears you, IndicMaxxer-2B answers, Mithi speaks. Headphones make interrupting work best.</p>
        </motion.div>
      </section>
    </main>
  );
}
