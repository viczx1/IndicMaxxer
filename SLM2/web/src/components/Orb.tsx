"use client";
import { useEffect, useRef } from "react";

export type OrbState = "idle" | "listening" | "thinking" | "speaking";

type Props = { state: OrbState; level: React.MutableRefObject<number>; size?: number; interactive?: boolean; onBoop?: () => void };

/** Glossy green glass orb with an orange orbit ring and floor ripples.
 *  `level` (0..1) drives it from audio. With `interactive`: highlight + liquid follow the cursor, click = squash + shockwave,
 *  drag sideways = spin the ring. */
export default function Orb({ state, level, size = 420, interactive = false, onBoop }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const stRef = useRef(state); stRef.current = state;
  const boopRef = useRef(onBoop); boopRef.current = onBoop;
  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext("2d")!;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let raf = 0, sm = 0, spin = 0;
    const it = { x: 0, y: 0, tx: 0, ty: 0, hover: 0, hv: 0, sq: 0, sv: 0, spinV: 0, drag: false, lastX: 0, moved: 0, bursts: [] as number[] };
    const fit = () => { const s = cv.clientWidth; cv.width = s * dpr; cv.height = s * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); };
    fit(); window.addEventListener("resize", fit);
    const t0 = performance.now();
    const local = (e: PointerEvent) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width - 0.5, (e.clientY - r.top) / r.height - 0.44]; };
    const onMove = (e: PointerEvent) => {
      const [x, y] = local(e); it.tx = Math.max(-1, Math.min(1, x * 2.2)); it.ty = Math.max(-1, Math.min(1, y * 2.2));
      it.hover = Math.hypot(x, y) < 0.3 ? 1 : 0.35;
      if (it.drag) { const dx = e.clientX - it.lastX; it.spinV += dx * 0.0016; it.moved += Math.abs(dx); it.lastX = e.clientX; }
    };
    const onDown = (e: PointerEvent) => { it.drag = true; it.lastX = e.clientX; it.moved = 0; cv.setPointerCapture(e.pointerId); };
    const onUp = (e: PointerEvent) => {
      it.drag = false;
      if (it.moved < 8) { it.sv -= 0.55; it.bursts.push(performance.now()); it.spinV += 0.12; boopRef.current?.(); }
      try { cv.releasePointerCapture(e.pointerId); } catch {}
    };
    const onLeave = () => { it.tx = 0; it.ty = 0; it.hover = 0; };
    if (interactive) {
      cv.addEventListener("pointermove", onMove); cv.addEventListener("pointerdown", onDown);
      cv.addEventListener("pointerup", onUp); cv.addEventListener("pointerleave", onLeave);
    }
    const ring = (cx: number, cy: number, R: number, front: boolean, glow: number) => {
      ctx.save(); ctx.translate(cx, cy); ctx.rotate(-0.18 + it.x * 0.08);
      const rx = R * 1.42, ry = R * (0.26 + it.y * 0.06);
      const a0 = front ? 0 : Math.PI, a1 = front ? Math.PI : Math.PI * 2;
      for (const [lw, al] of [[Math.max(10, R * 0.09), 0.22 * glow], [Math.max(5, R * 0.045), 0.55 * glow], [Math.max(2.5, R * 0.02), 1]] as const) {
        const g = ctx.createLinearGradient(-rx, 0, rx, 0);
        g.addColorStop(0, `rgba(255,120,20,${0.1 * al})`); g.addColorStop(0.3, `rgba(255,150,40,${al})`);
        g.addColorStop(0.7, `rgba(255,195,95,${al})`); g.addColorStop(1, `rgba(255,120,20,${0.1 * al})`);
        ctx.strokeStyle = g; ctx.lineWidth = lw; ctx.lineCap = "round";
        ctx.beginPath(); ctx.ellipse(0, 0, rx, Math.max(2, ry), 0, a0, a1); ctx.stroke();
      }
      const sa = ((spin % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
      if ((sa < Math.PI) === front) {
        const sx = Math.cos(sa) * rx, sy = Math.sin(sa) * ry, sr = 14 + Math.min(20, Math.abs(it.spinV) * 300);
        const sg = ctx.createRadialGradient(sx, sy, 0, sx, sy, sr);
        sg.addColorStop(0, "rgba(255,245,210,0.95)"); sg.addColorStop(1, "rgba(255,140,30,0)");
        ctx.fillStyle = sg; ctx.beginPath(); ctx.arc(sx, sy, sr, 0, 6.283); ctx.fill();
      }
      ctx.restore();
    };
    let prev = performance.now();
    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - prev) / 1000); prev = now;
      const f = dt * 60;                                       // frame-rate independent (60/120/144 Hz)
      const t = (now - t0) / 1000, s = cv.clientWidth, cx = s / 2, cy = s * 0.44;
      const st = stRef.current;
      it.x += (it.tx - it.x) * (1 - Math.pow(0.86, f)); it.y += (it.ty - it.y) * (1 - Math.pow(0.86, f));
      it.hv += (it.hover - it.hv) * (1 - Math.pow(0.9, f));
      it.sv += (-0.09 * it.sq - 0.16 * it.sv) * f; it.sq += it.sv * f * 0.5;          // squash spring
      it.spinV *= Math.pow(0.965, f);
      const target = (st === "thinking" ? 0.18 + 0.1 * Math.sin(t * 4) : st === "idle" ? 0.04 + 0.03 * Math.sin(t * 1.3) : level.current) + it.hv * 0.12;
      sm += (target - sm) * (1 - Math.pow(0.8, f));
      spin += ((st === "thinking" ? 0.06 : st === "idle" ? 0.012 : 0.025) + it.spinV) * f;
      const R = s * 0.24 * (1 + sm * 0.1);
      const bob = Math.sin(t * 1.2) * s * 0.008 + it.y * s * 0.01;
      const ox = it.x * s * 0.012;
      ctx.clearRect(0, 0, s, s);
      const fy = cy + R * 1.18;
      for (let i = 0; i < 4; i++) {
        const p = (t * (st === "listening" ? 0.6 : 0.35) + i / 4) % 1;
        ctx.strokeStyle = `rgba(200,255,170,${(1 - p) * (0.35 + sm * 0.8)})`; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.ellipse(cx, fy, R * (0.7 + p * 1.1), R * (0.13 + p * 0.2), 0, 0, 6.283); ctx.stroke();
      }
      const pool = ctx.createRadialGradient(cx, fy, 0, cx, fy, R * 1.1);
      pool.addColorStop(0, `rgba(170,255,120,${0.35 + sm * 0.4})`); pool.addColorStop(1, "rgba(170,255,120,0)");
      ctx.fillStyle = pool; ctx.beginPath(); ctx.ellipse(cx, fy, R * 1.1, R * 0.2, 0, 0, 6.283); ctx.fill();
      const y = cy + bob, x = cx + ox;
      const halo = ctx.createRadialGradient(x, y, R * 0.9, x, y, R * 1.8);
      halo.addColorStop(0, `rgba(190,255,130,${0.35 + sm * 0.5})`); halo.addColorStop(1, "rgba(190,255,130,0)");
      ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(x, y, R * 1.8, 0, 6.283); ctx.fill();
      // shockwaves from clicks
      it.bursts = it.bursts.filter((b) => now - b < 900);
      for (const b of it.bursts) {
        const p = (now - b) / 900;
        ctx.strokeStyle = `rgba(230,255,200,${(1 - p) * 0.8})`; ctx.lineWidth = 3 * (1 - p) + 0.5;
        ctx.beginPath(); ctx.arc(x, y, R * (1 + p * 0.9), 0, 6.283); ctx.stroke();
      }
      ring(x, y, R, false, 0.7 + sm);
      // sphere body (squash & stretch)
      ctx.save(); ctx.translate(x, y); ctx.scale(1 - it.sq * 0.12, 1 + it.sq * 0.12); ctx.translate(-x, -y);
      const body = ctx.createRadialGradient(x - R * 0.1 + it.x * R * 0.15, y + R * 0.45, R * 0.05, x, y, R * 1.02);
      body.addColorStop(0, "#e4ff86"); body.addColorStop(0.25, "#86e040"); body.addColorStop(0.62, "#2f9e2c");
      body.addColorStop(0.9, "#0f5418"); body.addColorStop(1, "#083a10");
      ctx.fillStyle = body; ctx.beginPath(); ctx.arc(x, y, R, 0, 6.283); ctx.fill();
      ctx.save(); ctx.beginPath(); ctx.arc(x, y, R, 0, 6.283); ctx.clip(); ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < 3; i++) {
        const ang = spin * (i % 2 ? -0.8 : 1) + i * 2.1;
        const bx = x + Math.cos(ang) * R * 0.38 + it.x * R * 0.35, by = y + Math.sin(ang * 1.2) * R * 0.3 + R * 0.15 + it.y * R * 0.25;
        const g = ctx.createRadialGradient(bx, by, 0, bx, by, R * (0.55 + sm * 0.4));
        g.addColorStop(0, `rgba(210,255,120,${0.28 + sm * 0.5})`); g.addColorStop(1, "rgba(210,255,120,0)");
        ctx.fillStyle = g; ctx.fillRect(x - R, y - R, R * 2, R * 2);
      }
      ctx.globalCompositeOperation = "source-over";
      const rim = ctx.createRadialGradient(x, y, R * 0.82, x, y, R);
      rim.addColorStop(0, "rgba(255,255,255,0)"); rim.addColorStop(1, "rgba(235,255,215,0.5)");
      ctx.fillStyle = rim; ctx.fillRect(x - R, y - R, R * 2, R * 2);
      const gx = x - R * 0.05 - it.x * R * 0.18, gy = y - R * 0.5 - it.y * R * 0.12;     // highlight leans away from cursor
      const gl = ctx.createLinearGradient(0, gy - R * 0.44, 0, gy + R * 0.44);
      gl.addColorStop(0, "rgba(255,255,255,0.9)"); gl.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = gl; ctx.beginPath(); ctx.ellipse(gx, gy, R * 0.68, R * 0.44, -0.15 - it.x * 0.2, 0, 6.283); ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,0.92)"; ctx.beginPath(); ctx.ellipse(gx - R * 0.37, gy - R * 0.02, R * 0.09, R * 0.05, -0.6, 0, 6.283); ctx.fill();
      ctx.restore(); ctx.restore();
      ring(x, y, R, true, 0.7 + sm);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf); window.removeEventListener("resize", fit);
      cv.removeEventListener("pointermove", onMove); cv.removeEventListener("pointerdown", onDown);
      cv.removeEventListener("pointerup", onUp); cv.removeEventListener("pointerleave", onLeave);
    };
  }, [level, interactive]);
  return <canvas ref={ref} style={{ width: `min(78vw, ${size}px)`, touchAction: "none" }}
                 className={`aspect-square ${interactive ? "cursor-grab active:cursor-grabbing" : ""}`} />;
}

/** Waveform bars under the orb, driven by the same level ref. */
export function Wave({ level, active }: { level: React.MutableRefObject<number>; active: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const on = useRef(active); on.current = active;
  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext("2d")!;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const N = 64, hist = new Array(N).fill(0);
    let raf = 0, acc = 0, prev = performance.now();
    const fit = () => { cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); };
    fit(); window.addEventListener("resize", fit);
    const frame = (now: number) => {
      const w = cv.clientWidth, h = cv.clientHeight;
      acc += now - prev; prev = now;
      while (acc > 33) { acc -= 33; hist.shift(); hist.push(on.current ? level.current * (0.6 + Math.random() * 0.6) : 0.02); }
      ctx.clearRect(0, 0, w, h);
      const bw = w / N;
      for (let i = 0; i < N; i++) {
        const edge = Math.sin((i / (N - 1)) * Math.PI);
        const v = Math.max(0.03, Math.min(1, hist[i] * 1.6)) * edge;
        const bh = v * h * 0.9;
        const g = ctx.createLinearGradient(0, h / 2 - bh / 2, 0, h / 2 + bh / 2);
        g.addColorStop(0, "rgba(210,255,150,0.2)"); g.addColorStop(0.5, "rgba(180,255,110,0.95)"); g.addColorStop(1, "rgba(210,255,150,0.2)");
        ctx.fillStyle = g;
        ctx.fillRect(i * bw + bw * 0.3, h / 2 - bh / 2, Math.max(1.5, bw * 0.38), bh);
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", fit); };
  }, [level]);
  return <canvas ref={ref} className="h-16 w-full max-w-lg" aria-hidden />;
}
