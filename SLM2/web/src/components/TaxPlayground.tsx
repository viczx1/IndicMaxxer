"use client";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { API } from "@/lib/api";

type Res = { name: string; tokens: number; per_word: number; pieces: string[] };
const SAMPLES = [
  { label: "ಕನ್ನಡ", text: "ನಮ್ಮ ಊರಿನ ಹಬ್ಬಕ್ಕೆ ನೀವೂ ಬನ್ನಿ, ಊಟ ಮಾಡಿಕೊಂಡು ಹೋಗಿ." },
  { label: "हिंदी", text: "कल सुबह दस बजे बैंक में मीटिंग है, कृपया समय पर पहुँचें।" },
  { label: "Kanglish", text: "naale office ge bega banni, meeting ide guru" },
  { label: "Hinglish", text: "bhai kal ka plan kya hai, movie chalein kya?" },
  { label: "English", text: "Please send me the invoice before Friday evening." },
];
const ease = [0.16, 1, 0.3, 1] as const;
const TINTS = ["bg-sky/25 ring-sky/40", "bg-lime/25 ring-lime/40", "bg-amber-300/25 ring-amber-200/40", "bg-fuchsia-300/20 ring-fuchsia-200/35"];

export default function TaxPlayground() {
  const [text, setText] = useState(SAMPLES[0].text);
  const [res, setRes] = useState<Res[] | null>(null);
  const [words, setWords] = useState(0);
  const [view, setView] = useState(0);
  const [err, setErr] = useState(false);
  const [retry, setRetry] = useState(0);
  const [squeeze, setSqueeze] = useState(false);
  const [short, setShort] = useState<string | null>(null);

  useEffect(() => {
    let retryId: ReturnType<typeof setTimeout> | undefined;
    const id = setTimeout(() => {
      fetch(`${API}/api/tax`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, compress: squeeze }) })
        .then((r) => r.json()).then((d) => {
          setRes(d.results); setWords(d.words); setErr(false); setShort(d.compressed ?? null);
          if (!d.results?.length) retryId = setTimeout(() => setRetry((n) => n + 1), 1500);   // tokenizers still loading
        })
        .catch(() => { setErr(true); retryId = setTimeout(() => setRetry((n) => n + 1), 3000); });
    }, retry ? 0 : 220);
    return () => { clearTimeout(id); if (retryId) clearTimeout(retryId); };
  }, [text, retry, squeeze]);

  const sorted = useMemo(() => (res ? [...res].sort((a, b) => b.tokens - a.tokens) : []), [res]);
  const ours = res?.find((r) => r.name.startsWith("IndicMaxxer"))?.tokens ?? 1;
  const max = Math.max(1, ...(res ?? []).map((r) => r.tokens));
  const shown = sorted[Math.min(view, Math.max(0, sorted.length - 1))];

  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_1.15fr]">
      <div>
        <div className="kicker mb-2">try it · type anything</div>
        <h2 className="text-3xl font-semibold sm:text-4xl">See the token tax live.</h2>
        <p className="mt-3 text-[var(--muted)]">Every token costs money and time, and uses up the context window. Same text through stock Sarvam-1 and IndicMaxxer (Sarvam-1 + our fused tokenizer + compressor).</p>
        <div className="mt-4 grid grid-cols-5 gap-1.5 text-center">
          {[["Hinglish", "−12.5%"], ["Kanglish", "−8.3%"], ["English", "−6.3%"], ["ಕನ್ನಡ", "−6.1%"], ["हिंदी", "−2.1%"]].map(([l, v]) => (
            <div key={l} className="glass-dark py-2"><div className="display text-lg font-semibold text-lime">{v}</div><div className="text-[11px] text-[var(--muted)]">{l}</div></div>
          ))}
        </div>
        <p className="mt-1.5 text-xs text-[var(--faint)]">tokens vs stock Sarvam-1 from the fused tokenizer alone (300 chat replies per style). The compressor cuts another 40–60% of long context.</p>
        <div className="mt-5 flex flex-wrap gap-2">
          {SAMPLES.map((s) => (
            <button key={s.label} onClick={() => setText(s.text)} className={`pill px-4 py-1.5 text-sm ${text === s.text ? "pill-on" : ""}`}>{s.label}</button>
          ))}
        </div>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} data-lenis-prevent
                  className="glass-dark mt-4 w-full resize-none p-4 text-lg leading-relaxed outline-none transition focus:ring-2 focus:ring-sky/60" />
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <span className="kicker">{words} words</span>
          <button onClick={() => setSqueeze(!squeeze)} className={`pill px-4 py-1.5 text-sm ${squeeze ? "pill-green" : ""}`}>
            {squeeze ? "Compressor on" : "Turn on the token compressor"}
          </button>
        </div>
        <AnimatePresence>
          {squeeze && short && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
              <div className="glass-dark mt-3 p-3 text-[15px] leading-relaxed"><span className="kicker mr-2">what the model reads:</span>{short}</div>
              <p className="mt-2 text-xs text-[var(--faint)]">LLMLingua-2 drops the words that carry the least information. Built for long documents, search results and chat history, not one-line questions.</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      <div className="space-y-5">
        {err && <p className="text-amber-100">Start the backend to measure live (it only needs the tokenizers, not the model).</p>}
        {sorted.map((r) => {
          const me = r.name.startsWith("IndicMaxxer");
          return (
            <div key={r.name}>
              <div className="mb-1.5 flex items-baseline justify-between gap-3">
                <span className={me ? "font-semibold" : "text-[var(--muted)]"}>{r.name}</span>
                <span className="tabular-nums text-[var(--muted)]">
                  <motion.b key={r.tokens} initial={{ opacity: 0.3, y: -4 }} animate={{ opacity: 1, y: 0 }} className="display text-2xl text-white">{r.tokens}</motion.b> tokens
                  <span className="ml-2 text-sm">{r.per_word}/word{!me && ` · ${(r.tokens / ours).toFixed(1)}×`}</span>
                </span>
              </div>
              <div className="glass-dark h-4 overflow-hidden !rounded-full p-[3px]">
                <motion.div animate={{ width: `${(r.tokens / max) * 100}%` }} transition={{ duration: 0.7, ease }}
                            className={`relative h-full rounded-full ${r.name.includes("compressor") ? "bg-gradient-to-b from-[#b9fff0] to-[#14b8a6]" : me ? "bg-gradient-to-b from-[#c4f58a] to-[#3fae3a]" : r.name.startsWith("GPT") ? "bg-gradient-to-b from-[#ffc27a] to-[#e8681a]" : "bg-gradient-to-b from-[#a9e8ff] to-[#2b8fd0]"}`}>
                  <span className="absolute inset-x-1 top-[1px] h-[40%] rounded-full bg-white/50" />
                </motion.div>
              </div>
            </div>
          );
        })}
        {shown && (
          <div className="glass-inset p-4">
            <div className="mb-3 flex flex-wrap gap-1.5">
              {sorted.map((r, i) => (
                <button key={r.name} onClick={() => setView(i)} className={`rounded-full px-3 py-1 text-xs transition ${i === view ? "bg-white/30 text-white" : "text-[var(--muted)] hover:bg-white/15"}`}>{r.name}</button>
              ))}
            </div>
            <div className="flex max-h-40 flex-wrap gap-1 overflow-y-auto thin-scroll" data-lenis-prevent>
              <AnimatePresence initial={false}>
                {shown.pieces.map((p, i) => (
                  <motion.span key={`${shown.name}-${i}-${p}`} initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                               transition={{ duration: 0.25, delay: Math.min(i * 0.008, 0.4) }}
                               className={`rounded-md px-1.5 py-0.5 text-sm ring-1 ${TINTS[i % TINTS.length]}`}>
                    {p.trim() === "" ? "␣" : p.replace(/�/g, "�")}
                  </motion.span>
                ))}
              </AnimatePresence>
            </div>
            <p className="kicker mt-3">each chip is one token</p>
          </div>
        )}
      </div>
    </div>
  );
}
