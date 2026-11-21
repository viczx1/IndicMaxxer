"use client";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { AudioLines, Keyboard, Mic, Square, Globe, ChevronDown, User, Repeat, Send, Fingerprint, X } from "lucide-react";
import Aurora from "@/components/Aurora";
import Nav from "@/components/Nav";
import Orb, { OrbState, Wave } from "@/components/Orb";
import { Msg, enrollVoice, resetVoiceprint, streamChat, stt, sttWarm, toWav16k, tts, voiceprintStatus } from "@/lib/api";

// Spoken Kanglish is spoken Kannada (same for Hinglish/Hindi): the Kannada / Hindi recogniser handles it.
const LANGS = [
  { id: "kn", label: "ಕನ್ನಡ · Kanglish", name: "Kannada", read: "ನಮಸ್ಕಾರ, ನಾನು ಇಂಡಿಕ್‌ಮ್ಯಾಕ್ಸರ್ ಜೊತೆ ಮಾತಾಡುತ್ತಿದ್ದೇನೆ. ಇದು ನನ್ನ ಧ್ವನಿ, ಬೇರೆ ಯಾರದ್ದೂ ಅಲ್ಲ." },
  { id: "hi", label: "हिंदी · Hinglish", name: "Hindi", read: "नमस्ते, मैं इंडिकमैक्सर से बात कर रहा हूँ। यह मेरी आवाज़ है, किसी और की नहीं।" },
  { id: "en", label: "English", name: "English", read: "Hi, I'm talking to IndicMaxxer. This is my voice, and only my voice should count." },
];
const VOICES = [{ id: "charvi", label: "Charvi" }, { id: "satvik", label: "Satvik" }];
const STATUS: Record<OrbState, string> = { idle: "Tap the mic and speak", listening: "Your turn, go ahead…", thinking: "Thinking…", speaking: "Speaking… (tap the orb to cut in)" };
const ease = [0.16, 1, 0.3, 1] as const;
type Turn = { you: string; bot: string; cut?: boolean };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Vad = any;

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
  const [note, setNote] = useState("");
  const [loop, setLoop] = useState(true);
  const [onlyMe, setOnlyMe] = useState(true);
  const [enrolled, setEnrolled] = useState(false);
  const [enrolling, setEnrolling] = useState<"" | "ready" | "recording" | "saving">("");
  const level = useRef(0);
  const actx = useRef<AudioContext | null>(null);
  const vad = useRef<Vad>(null);
  const history = useRef<Msg[]>([]);
  const cancel = useRef<() => void>(() => {});
  const stateRef = useRef(state); stateRef.current = state;
  const loopRef = useRef(loop); loopRef.current = loop;
  const textRef = useRef(textMode); textRef.current = textMode;
  const langRef = useRef(lang); langRef.current = lang;
  const voiceRef = useRef(voice); voiceRef.current = voice;
  const onlyMeRef = useRef(onlyMe); onlyMeRef.current = onlyMe;
  const enrollRef = useRef(enrolling); enrollRef.current = enrolling;
  const enrollRef2 = useRef(false);
  const feed = useRef<HTMLDivElement>(null);

  useEffect(() => { feed.current?.scrollTo({ top: feed.current.scrollHeight, behavior: "smooth" }); }, [turns, phase]);
  useEffect(() => { sttWarm(lang.id); }, [lang]);
  useEffect(() => { resetVoiceprint().catch(() => {}); setEnrolled(false); void voiceprintStatus; }, []);   // lock on per session
  const loud = useRef(0);          // running loudness of the main speaker

  const ctx = () => (actx.current ??= new AudioContext());
  // speaking: only clear, sustained speech counts as an interruption (the speaker's own echo shouldn't)
  const setSensitivity = (speaking: boolean) => vad.current?.setOptions(speaking
    ? { positiveSpeechThreshold: 0.7, negativeSpeechThreshold: 0.45, minSpeechMs: 300 }
    : { positiveSpeechThreshold: 0.6, negativeSpeechThreshold: 0.35, minSpeechMs: 300 });

  /** Silero VAD (neural speech detector, runs in the browser): fans, keyboards, traffic and music aren't speech. */
  async function ensureVad() {
    if (vad.current) return;
    const { MicVAD } = await import("@ricky0123/vad-web");
    vad.current = await MicVAD.new({
      model: "v5", baseAssetPath: "/vad/", onnxWASMBasePath: "/vad/",
      positiveSpeechThreshold: 0.5, negativeSpeechThreshold: 0.3, redemptionMs: 1500, preSpeechPadMs: 1000, minSpeechMs: 300,
      getStream: () => navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } }),
      onFrameProcessed: (p: { isSpeech: number }, frame: Float32Array) => {
        if (stateRef.current !== "listening" && !enrollRef.current) return;
        let s = 0; for (const v of frame) s += v * v;
        level.current = Math.min(1, Math.sqrt(s / frame.length) * 6 * (0.4 + p.isSpeech));
      },
      onSpeechStart: () => { if (stateRef.current === "listening") { setPhase("hearing"); if (sendTimer.current) clearTimeout(sendTimer.current); } },
      onSpeechRealStart: () => {},   // half-duplex: tap the orb to interrupt (talking over speakers gets garbled)
      onVADMisfire: () => {
        if (stateRef.current !== "listening") return;
        if (pieces.current.length) onSegment(new Float32Array(0)); else setPhase("");   // re-arm the send timer
      },
      onSpeechEnd: (audio: Float32Array) => onSegment(audio),
    });
    vad.current.start();
  }

  // Silero splits a sentence at every pause: collect pieces and only send once you've been quiet for a while
  const pieces = useRef<Float32Array[]>([]);
  const sendTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  function onSegment(audio: Float32Array) {
    if (stateRef.current !== "listening") { pieces.current = []; return; }
    pieces.current.push(audio);
    if (sendTimer.current) clearTimeout(sendTimer.current);
    sendTimer.current = setTimeout(() => {
      const all = pieces.current; pieces.current = [];
      const n = all.reduce((a, c) => a + c.length, 0), joined = new Float32Array(n); let o = 0;
      for (const c of all) { joined.set(c, o); o += c.length; }
      if (n) onUtterance(joined);
    }, 1300);
  }
  async function onUtterance(audio: Float32Array) {
    const wav = toWav16k([audio], 16000);
    if (enrollRef.current === "recording") { finishEnroll(wav, audio.length / 16000); return; }
    if (stateRef.current !== "listening") return;
    let e2 = 0; for (const v of audio) e2 += v * v; const rms = Math.sqrt(e2 / audio.length);
    if (onlyMeRef.current && loud.current > 0 && rms < loud.current * 0.45) {     // much quieter = background voice
      setPhase(""); setNote("Ignored a quieter background voice"); setTimeout(() => setNote(""), 2000); return;
    }
    setPhase("transcribing"); setState("thinking");
    try {
      const r = await stt(wav, langRef.current.id, onlyMeRef.current && enrollRef2.current);
      setPhase("");
      if (r.rejected) {                                   // someone else / TV: ignore and keep listening
        setNote(`Ignored a voice that isn't yours (match ${Math.round((r.similarity ?? 0) * 100)}%)`);
        setTimeout(() => setNote(""), 2500);
        setState("listening"); return;
      }
      if (r.text.trim()) {
        loud.current = loud.current ? loud.current * 0.7 + rms * 0.3 : rms;
        if (!enrollRef2.current && audio.length / 16000 >= 1.5) {   // auto-lock onto the main speaker's voice (WavLM)
          enrollRef2.current = true; enrollVoice(wav, true).then(() => setEnrolled(true)).catch(() => { enrollRef2.current = false; });
        }
        think(r.text.trim());
      }
      else setState(loopRef.current ? "listening" : "idle");
    } catch {
      setPhase(""); setState("idle");
      setErr("Speech recognition failed. Is the backend running?");
    }
  }

  async function listen() {
    setErr("");
    try { await ensureVad(); } catch (e) { setErr(`Microphone: ${(e as Error).message}`); return; }
    ctx().resume(); setSensitivity(false);
    setPhase(""); setState("listening");
  }

  function interrupt() {
    if (stateRef.current !== "speaking" && stateRef.current !== "thinking") return;
    cancel.current();
    setTurns((t) => t.length ? [...t.slice(0, -1), { ...t[t.length - 1], cut: true }] : t);
    setSensitivity(false); setState("listening"); setPhase("hearing");   // VAD keeps recording this utterance
  }

  async function think(text: string) {
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
      if (queue.length === 0) {
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
        if (stateRef.current !== "speaking") { setSensitivity(true); setState("speaking"); }
        meter();
        await new Promise<void>((res) => { src.onended = () => res(); src.start(); });
        on = false; level.current = 0;
      }
    })();
    try {
      await streamChat({ message: text, history: history.current, mode: "instruct", k: 1, forceSearch: false, style: langRef.current.id, voice: true }, (e) => {
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
    if (!stopped) { setSensitivity(false); setState(loopRef.current && !textRef.current ? "listening" : "idle"); }
  }

  // ---- only-my-voice enrolment
  async function startEnroll() {
    cancel.current(); setState("idle"); setErr("");
    try { await ensureVad(); } catch (e) { setErr(`Microphone: ${(e as Error).message}`); return; }
    ctx().resume(); setSensitivity(false); setEnrolling("ready");
  }
  async function finishEnroll(wav: Blob, secs: number) {
    if (secs < 2) { setNote("That was short: read the whole sentence."); setTimeout(() => setNote(""), 2500); return; }
    setEnrolling("saving");
    const r = await enrollVoice(wav, !enrolled).catch(() => ({ ok: false, error: "backend unreachable" }));
    if (r.ok) { setEnrolled(true); setOnlyMe(true); setEnrolling(""); setNote("Voice saved. It will now ignore other voices."); setTimeout(() => setNote(""), 3000); }
    else { setEnrolling("ready"); setErr(r.error ?? "Could not save your voice."); }
  }

  function micButton() {
    ctx().resume();
    if (state === "idle") listen();
    else if (state === "listening") { setState("idle"); setPhase(""); }
    else interrupt();
  }
  function stop() { cancel.current(); setState("idle"); setPhase(""); level.current = 0; setLoop(false); }
  function sendTyped() {
    const t = typed.trim(); if (!t) return;
    cancel.current(); ctx().resume(); setTyped(""); think(t);
  }

  useEffect(() => () => { cancel.current(); vad.current?.destroy?.(); actx.current?.close(); }, []);

  const status = enrolling ? "Teaching it your voice…" : phase === "hearing" ? "Hearing you…" : phase === "transcribing" ? "Understanding…" : STATUS[state];
  return (
    <main className="relative min-h-screen">
      <Aurora calm />
      <Nav />
      <section className="relative z-10 mx-auto flex min-h-screen max-w-6xl items-center px-4 pb-8 pt-24">
        <motion.div initial={{ opacity: 0, y: 30, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.9, ease }}
                    className="glass w-full p-5 sm:p-8">
          <div className="grid items-center gap-6 lg:grid-cols-[1.15fr_1fr]">
            <div className="flex flex-col items-center">
              <div onClick={micButton} className="cursor-pointer"><Orb state={enrolling ? "listening" : state} level={level} /></div>
              <AnimatePresence mode="wait">
                <motion.div key={status} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.25 }}
                            className="display soft-glow -mt-6 text-center text-2xl font-light sm:text-3xl">{status}</motion.div>
              </AnimatePresence>
              <Wave level={level} active={state === "listening" || state === "speaking" || !!enrolling} />
              <AnimatePresence>{note && <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="kicker mt-1">{note}</motion.p>}</AnimatePresence>
            </div>
            <div className="glass-inset relative flex h-[420px] flex-col p-3">
              <AnimatePresence>
                {enrolling && (
                  <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.97 }}
                              className="absolute inset-0 z-10 grid place-items-center rounded-[14px] bg-[rgba(6,60,70,0.55)] p-6 text-center backdrop-blur-md">
                    <div>
                      <button onClick={() => setEnrolling("")} className="absolute right-3 top-3 text-white/70 hover:text-white"><X size={18} /></button>
                      <Fingerprint className="mx-auto mb-3 text-lime" size={34} />
                      <p className="text-lg">Read this aloud, at your normal volume:</p>
                      <p className="glass-dark mx-auto mt-4 max-w-md p-4 text-xl leading-relaxed">{lang.read}</p>
                      <p className="mt-4 text-sm text-[var(--muted)]">{enrolling === "saving" ? "Saving your voiceprint…" : "Start whenever you're ready. It stops by itself when you finish."}</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
              <div ref={feed} data-lenis-prevent className="thin-scroll flex-1 space-y-3 overflow-y-auto p-1">
                {turns.length === 0 && (
                  <div className="grid h-full place-items-center px-6 text-center text-[var(--muted)]">
                    <div>
                      <p className="text-lg text-white">Talk to IndicMaxxer in {lang.name}.</p>
                      <p className="mt-2 text-sm">Speak naturally, it waits until you finish. It locks onto your voice automatically and ignores quieter background voices. Talk over it to interrupt.</p>

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
              <button onClick={() => setTextMode(false)} className={`pill px-5 py-2.5 ${!textMode ? "pill-on" : ""}`}><AudioLines size={18} /> Voice</button>
              <button onClick={() => { setTextMode(true); if (state === "listening") { setState("idle"); setPhase(""); } }}
                      className={`pill px-5 py-2.5 ${textMode ? "pill-on" : ""}`}><Keyboard size={18} /> Text</button>
            </div>
            {textMode ? (
              <div className="flex min-w-[240px] flex-1 items-center gap-2 sm:max-w-sm">
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
                <button onClick={() => setLangOpen(!langOpen)} className="pill px-4 py-2.5"><Globe size={18} /> {lang.label} <ChevronDown size={16} className={`transition ${langOpen ? "rotate-180" : ""}`} /></button>
                <AnimatePresence>
                  {langOpen && (
                    <motion.ul initial={{ opacity: 0, y: 8, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 8, scale: 0.97 }}
                               transition={{ duration: 0.18 }} className="glass absolute bottom-14 left-0 z-20 w-60 !rounded-2xl p-1.5">
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
              <button onClick={() => setOnlyMe(!onlyMe)} title="Focus on the main (loudest) speaker and ignore background voices"
                      className={`pill px-4 py-2.5 ${onlyMe ? "pill-on" : ""}`}><Fingerprint size={17} /> {onlyMe ? (enrolled ? "Locked on you" : "Focus: main voice") : "Anyone"}</button>
              <button onClick={() => setLoop(!loop)} title="Conversation mode: keeps listening after each reply" className={`pill px-4 py-2.5 ${loop ? "pill-on" : ""}`}><Repeat size={17} /></button>
              <button onClick={stop} className="pill px-4 py-2.5"><Square size={16} className="fill-white" /> Stop</button>
            </div>
          </div>
          <p className="mt-4 text-center text-xs text-[var(--faint)]">Runs fully on your laptop: Silero detects speech, WavLM locks onto the main speaker, Whisper transcribes, IndicMaxxer-2B answers, Mithi speaks.</p>
        </motion.div>
      </section>
    </main>
  );
}
