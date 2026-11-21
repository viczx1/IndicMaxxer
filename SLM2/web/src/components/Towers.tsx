"use client";
import { useEffect, useRef } from "react";

/** PS2-boot-inspired field: translucent jade towers breathing on a dark floor, drifting motes, mouse parallax. */
export default function Towers({ intensity = 1 }: { intensity?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current!;
    const ctx = cv.getContext("2d")!;
    let w = 0, h = 0, raf = 0;
    const t0 = performance.now();
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const mouse = { x: 0, y: 0, tx: 0, ty: 0 };
    const N = 13, towers: { x: number; z: number; ph: number; sp: number; base: number }[] = [];
    for (let i = 0; i < N; i++)
      for (let j = 0; j < 9; j++)
        if (Math.random() < 0.42)
          towers.push({ x: (i - N / 2) * 1.6 + Math.random() * 0.6, z: 3 + j * 1.7 + Math.random() * 0.8,
                        ph: Math.random() * 6.28, sp: 0.15 + Math.random() * 0.35, base: 0.4 + Math.random() * 2.6 });
    towers.sort((a, b) => b.z - a.z);
    const motes = Array.from({ length: 90 }, () => ({ x: Math.random(), y: Math.random(), r: Math.random() * 1.6 + 0.3,
                                                       v: 0.004 + Math.random() * 0.012, a: Math.random() }));
    const resize = () => {
      w = cv.clientWidth; h = cv.clientHeight;
      cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const onMove = (e: PointerEvent) => { mouse.tx = e.clientX / w - 0.5; mouse.ty = e.clientY / h - 0.5; };
    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", onMove);

    const frame = (now: number) => {
      const t = (now - t0) / 1000;
      mouse.x += (mouse.tx - mouse.x) * 0.04; mouse.y += (mouse.ty - mouse.y) * 0.04;
      ctx.clearRect(0, 0, w, h);
      const horizon = h * (0.52 + mouse.y * 0.04), f = Math.min(w, h) * 0.9;
      const camX = Math.sin(t * 0.05) * 1.2 + mouse.x * 2.2;
      // floor glow
      const g = ctx.createRadialGradient(w / 2, horizon, 10, w / 2, horizon, w * 0.7);
      g.addColorStop(0, `rgba(52,245,176,${0.10 * intensity})`); g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g; ctx.fillRect(0, horizon - h * 0.3, w, h);
      // floor grid lines
      ctx.strokeStyle = `rgba(95,230,255,${0.05 * intensity})`; ctx.lineWidth = 1;
      for (let z = 2; z < 20; z += 1.2) {
        const zz = z - ((t * 0.35) % 1.2);
        const y = horizon + (1.6 * f) / zz;
        if (y > h) continue;
        ctx.globalAlpha = Math.min(1, 6 / zz); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      ctx.globalAlpha = 1;
      // towers
      for (const tw of towers) {
        const z = tw.z, sc = f / z;
        const x = w / 2 + (tw.x - camX) * sc;
        const ht = (tw.base + Math.sin(t * tw.sp + tw.ph) * 0.8 + 0.9) * sc * 0.55;
        const yb = horizon + 1.6 * sc * 0.6, wd = sc * 0.42;
        if (x < -wd || x > w + wd) continue;
        const fade = Math.max(0, Math.min(1, (16 - z) / 12)) * intensity;
        const lg = ctx.createLinearGradient(0, yb - ht, 0, yb);
        lg.addColorStop(0, `rgba(157,255,217,${0.55 * fade})`);
        lg.addColorStop(0.15, `rgba(52,245,176,${0.22 * fade})`);
        lg.addColorStop(1, `rgba(10,80,70,${0.05 * fade})`);
        ctx.fillStyle = lg; ctx.fillRect(x - wd / 2, yb - ht, wd, ht);
        ctx.fillStyle = `rgba(220,255,240,${0.7 * fade})`; ctx.fillRect(x - wd / 2, yb - ht, wd, Math.max(1, sc * 0.02));
        // reflection
        const rg = ctx.createLinearGradient(0, yb, 0, yb + ht * 0.5);
        rg.addColorStop(0, `rgba(52,245,176,${0.10 * fade})`); rg.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = rg; ctx.fillRect(x - wd / 2, yb, wd, ht * 0.5);
      }
      // motes
      for (const m of motes) {
        m.y -= m.v * 0.06; if (m.y < -0.02) { m.y = 1.02; m.x = Math.random(); }
        const a = (0.3 + 0.7 * Math.abs(Math.sin(t * 0.6 + m.a * 6))) * 0.6 * intensity;
        ctx.fillStyle = `rgba(180,255,230,${a})`;
        ctx.beginPath(); ctx.arc(m.x * w + mouse.x * 20 * m.r, m.y * h, m.r, 0, 6.28); ctx.fill();
      }
      raf = requestAnimationFrame(frame);
    };
    const vis = () => { cancelAnimationFrame(raf); if (!document.hidden) raf = requestAnimationFrame(frame); };
    raf = requestAnimationFrame(frame);
    document.addEventListener("visibilitychange", vis);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", resize);
                   window.removeEventListener("pointermove", onMove); document.removeEventListener("visibilitychange", vis); };
  }, [intensity]);
  return <canvas ref={ref} className="pointer-events-none fixed inset-0 h-full w-full" aria-hidden />;
}
