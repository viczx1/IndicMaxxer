"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { health } from "@/lib/api";

const LINKS = [{ href: "/", label: "Home" }, { href: "/chat", label: "Chat" }, { href: "/voice", label: "Voice" }];

export default function Nav() {
  const path = usePathname();
  const [st, setSt] = useState<string>("checking");
  useEffect(() => {
    let alive = true;
    const tick = () => health().then((h) => alive && setSt(h.status)).catch(() => alive && setSt("offline"));
    tick(); const id = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  const label = st === "ready" ? "model online" : st === "loading" || st === "starting" ? "model loading" : st === "no_model" ? "no model" : "model offline";
  const dot = st === "ready" ? "bg-lime shadow-[0_0_10px_#8be04e]" : st === "loading" || st === "starting" ? "bg-amber-300 animate-pulse" : "bg-rose-300";
  return (
    <header className="fixed inset-x-0 top-0 z-40 px-4 pt-4">
      <nav className="glass mx-auto flex max-w-6xl items-center gap-2 !rounded-full py-2 pl-3 pr-2">
        <Link href="/" className="flex items-center gap-2.5 pr-3">
          <span className="bead h-7 w-7" />
          <span className="leading-none">
            <span className="block text-[17px] font-semibold tracking-tight">IndicMaxxer</span>
            <span className="block text-[10.5px] text-[var(--muted)]">your language, your machine</span>
          </span>
        </Link>
        <div className="ml-auto flex items-center gap-1">
          {LINKS.map((l) => {
            const on = path === l.href;
            return (
              <Link key={l.href} href={l.href} className={`relative rounded-full px-4 py-1.5 text-[15px] transition ${on ? "text-white" : "text-white/75 hover:text-white"}`}>
                {on && <motion.span layoutId="navpill" className="pill absolute inset-0 !gap-0" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
                <span className="relative">{l.label}</span>
              </Link>
            );
          })}
          <span className="ml-2 hidden items-center gap-2 rounded-full px-3 py-1.5 text-xs text-[var(--muted)] sm:flex">
            <span className={`h-2 w-2 rounded-full ${dot}`} /> {label}
          </span>
        </div>
      </nav>
    </header>
  );
}
