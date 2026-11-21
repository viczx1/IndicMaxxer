"use client";
import Link from "next/link";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Brain, Globe2, Languages, Cpu, ArrowRight, Mic, MousePointerClick } from "lucide-react";
import Aurora from "@/components/Aurora";
import Nav from "@/components/Nav";
import Orb from "@/components/Orb";
import TaxPlayground from "@/components/TaxPlayground";
import { Magnetic, Tilt } from "@/components/fx";
import { tts } from "@/lib/api";

const GREETINGS = [
  { text: "ನಮಸ್ಕಾರ", style: "kn" }, { text: "नमस्ते", style: "hi" }, { text: "Hello", style: "en" },
  { text: "Namaskara", style: "kanglish" }, { text: "Kya scene hai", style: "hinglish" },
];
const ease = [0.16, 1, 0.3, 1] as const;

function Boot({ onDone }: { onDone: () => void }) {
  useEffect(() => { const id = setTimeout(onDone, 1700); return () => clearTimeout(id); }, [onDone]);
  return (
    <motion.div className="fixed inset-0 z-50 grid place-items-center bg-[linear-gradient(165deg,#0e5f9a,#0f7f8c_45%,#148a63)]"
                exit={{ opacity: 0, scale: 1.04 }} transition={{ duration: 0.7, ease }}>
      <motion.span initial={{ scale: 0.2, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.9, ease }} className="bead h-16 w-16" />
      <motion.span initial={{ opacity: 0, letterSpacing: "0.6em" }} animate={{ opacity: 1, letterSpacing: "0.32em" }} transition={{ delay: 0.35, duration: 1, ease }}
                   className="display absolute mt-32 text-lg font-light text-white/90">INDICMAXXER</motion.span>
    </motion.div>
  );
}

const FEATURES = [
  { icon: Languages, title: "Five ways of speaking, never mixed", body: "Kannada, Hindi, English, Kanglish and Hinglish. It answers in the one you used, and corrects itself if it slips into another." },
  { icon: Brain, title: "Reasoning in your own language", body: "Switch to /reasoning and it works the problem out in Kannada or Hindi, three times over, then goes with the answer the passes agree on." },
  { icon: Globe2, title: "Checks facts instead of guessing", body: "Web search, reading PDFs and Word files, writing documents and reading scanned text. A simple rule-based router decides when." },
  { icon: Cpu, title: "Runs on an ordinary laptop", body: "Compressed to 4-bit, the 2B model needs about 2 GB of GPU memory. No API bill, and nothing you say leaves your machine." },
];

export default function Home() {
  const [booted, setBooted] = useState(false);
  const [g, setG] = useState(0);
  const [boops, setBoops] = useState(0);
  const idle = useRef(0);
  const audio = useRef<HTMLAudioElement | null>(null);
  useEffect(() => { try { if (sessionStorage.getItem("imx-booted")) setBooted(true); } catch {} }, []);
  useEffect(() => { const id = setInterval(() => setG((v) => (v + 1) % GREETINGS.length), 2600); return () => clearInterval(id); }, [boops]);
  const done = () => { setBooted(true); try { sessionStorage.setItem("imx-booted", "1"); } catch {} };
  const boop = () => {
    const next = (g + 1) % GREETINGS.length;
    setG(next); setBoops((b) => b + 1);
    tts(GREETINGS[next].text, GREETINGS[next].style).then((b) => {
      if (!b) return;
      audio.current?.pause(); audio.current = new Audio(URL.createObjectURL(b)); audio.current.play().catch(() => {});
    }).catch(() => {});
  };
  const show = (d: number) => ({ initial: { opacity: 0, y: 24 }, animate: booted ? { opacity: 1, y: 0 } : {}, transition: { duration: 0.9, delay: d, ease } });

  return (
    <main className="relative min-h-screen">
      <AnimatePresence>{!booted && <Boot onDone={done} />}</AnimatePresence>
      <Aurora />
      <Nav />

      <section className="relative z-10 mx-auto grid min-h-screen max-w-6xl items-center gap-4 px-4 pb-16 pt-28 lg:grid-cols-[1.15fr_1fr]">
        <div className="text-center lg:text-left">
          <motion.div {...show(0.05)} className="kicker mb-4">The Big Hack 2.0 · the token tax</motion.div>
          <motion.h1 {...show(0.15)} className="soft-glow reflect text-6xl font-semibold sm:text-7xl lg:text-[88px] lg:leading-[1]">IndicMaxxer</motion.h1>
          <motion.p {...show(0.3)} className="display mt-7 text-2xl font-light sm:text-[32px]">
            <span className="relative inline-grid h-[1.25em] overflow-hidden align-bottom">
              <AnimatePresence mode="popLayout" initial={false}>
                <motion.span key={g} initial={{ y: "100%", opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: "-100%", opacity: 0 }}
                             transition={{ duration: 0.55, ease }} className="col-start-1 row-start-1 font-semibold text-lime">{GREETINGS[g].text}</motion.span>
              </AnimatePresence>
            </span>, I speak your language.
          </motion.p>
          <motion.p {...show(0.42)} className="mx-auto mt-5 max-w-xl text-lg leading-relaxed text-[var(--muted)] lg:mx-0">
            Indian languages cost up to 12× more tokens than English on popular models. That means a bigger bill, slower replies and less room for your documents.
            IndicMaxxer is a 2B model that cuts that cost and runs on your own laptop.
          </motion.p>
          <motion.div {...show(0.55)} className="mt-9 flex flex-wrap justify-center gap-3 lg:justify-start">
            <Magnetic><Link href="/chat" className="pill pill-green px-7 py-3 text-[17px] font-semibold">Start chatting <ArrowRight size={18} /></Link></Magnetic>
            <Magnetic><Link href="/voice" className="pill px-7 py-3 text-[17px]"><Mic size={18} /> Talk to it</Link></Magnetic>
          </motion.div>
        </div>
        <motion.div initial={{ opacity: 0, scale: 0.85 }} animate={booted ? { opacity: 1, scale: 1 } : {}} transition={{ duration: 1.2, delay: 0.2, ease }}
                    className="flex flex-col items-center">
          <Orb state="idle" level={idle} size={480} interactive onBoop={boop} />
          <motion.div initial={{ opacity: 0 }} animate={booted ? { opacity: 1 } : {}} transition={{ delay: 1.4 }}
                      className="kicker -mt-8 flex items-center gap-2">
            <MousePointerClick size={14} /> {boops === 0 ? "tap the orb · drag to spin" : `it said ${GREETINGS[g].text}`}
          </motion.div>
        </motion.div>
      </section>

      <section className="relative z-10 mx-auto max-w-6xl space-y-8 px-4 pb-24">
        <motion.div initial={{ opacity: 0, y: 40 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-80px" }}
                    transition={{ duration: 0.9, ease }} className="glass p-7 sm:p-10">
          <TaxPlayground />
        </motion.div>

        <div className="grid gap-6 sm:grid-cols-2">
          {FEATURES.map((f, i) => (
            <motion.div key={f.title} initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-60px" }}
                        transition={{ duration: 0.8, delay: i * 0.07, ease }}>
              <Tilt className="h-full"><div className="glass h-full p-7">
                <div className="mb-5 grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-b from-white/45 to-white/10 ring-1 ring-white/60 shadow-[inset_0_1px_0_rgba(255,255,255,.8)]">
                  <f.icon size={22} className="text-white drop-shadow" />
                </div>
                <h3 className="text-xl font-semibold">{f.title}</h3>
                <p className="mt-2 leading-relaxed text-[var(--muted)]">{f.body}</p>
              </div></Tilt>
            </motion.div>
          ))}
        </div>

        <motion.div initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.9, ease }}
                    className="glass flex flex-col items-center gap-5 p-10 text-center">
          <h2 className="text-3xl font-semibold">Say something in your language.</h2>
          <div className="flex flex-wrap justify-center gap-3">
            <Magnetic><Link href="/chat" className="pill pill-green px-7 py-3 text-[17px] font-semibold">Open chat <ArrowRight size={18} /></Link></Magnetic>
            <Magnetic><Link href="/voice" className="pill px-7 py-3 text-[17px]"><Mic size={18} /> Open voice</Link></Magnetic>
          </div>
        </motion.div>
        <footer className="pt-4 text-center text-sm text-[var(--faint)]">Datamaxxers · built on Sarvam-1 (non-commercial licence) · runs on your machine</footer>
      </section>
    </main>
  );
}
