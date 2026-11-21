"use client";
import { useEffect } from "react";
import Lenis from "lenis";

/** Lenis inertial scrolling (frame-rate independent, so it runs at the display's 120/144 Hz)
 *  + a pointer-following Aero sheen on every .glass panel (CSS vars, no React re-renders). */
export default function SmoothScroll() {
  useEffect(() => {
    const lenis = new Lenis({ lerp: 0.11, smoothWheel: true, wheelMultiplier: 0.9 });
    let raf = 0;
    const loop = (t: number) => { lenis.raf(t); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    let last: HTMLElement | null = null;
    const move = (e: PointerEvent) => {
      const el = (e.target as HTMLElement)?.closest?.(".glass") as HTMLElement | null;
      if (last && last !== el) last.style.setProperty("--sheen", "0");
      if (el) {
        const r = el.getBoundingClientRect();
        el.style.setProperty("--mx", `${e.clientX - r.left}px`);
        el.style.setProperty("--my", `${e.clientY - r.top}px`);
        el.style.setProperty("--sheen", "1");
      }
      last = el;
    };
    window.addEventListener("pointermove", move, { passive: true });
    return () => { cancelAnimationFrame(raf); lenis.destroy(); window.removeEventListener("pointermove", move); };
  }, []);
  return null;
}
