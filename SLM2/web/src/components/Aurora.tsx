"use client";
import { useEffect, useRef } from "react";

/** Vista-style aurora: soft light ribbons sweeping across the page, rendered at half resolution for natural blur. */
export default function Aurora({ calm = false }: { calm?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext("2d")!;
    const SCALE = 0.34;
    let w = 0, h = 0, raf = 0;
    const t0 = performance.now();
    const mouse = { x: 0, y: 0, tx: 0, ty: 0 };
    const ribbons = [
      { y: 0.30, amp: 0.10, thick: 0.10, speed: 0.06, ph: 0.0, tilt: 0.38, color: [235, 255, 250], a: 0.42 },
      { y: 0.48, amp: 0.08, thick: 0.05, speed: 0.08, ph: 1.7, tilt: 0.30, color: [160, 240, 255], a: 0.35 },
      { y: 0.62, amp: 0.12, thick: 0.16, speed: 0.05, ph: 3.1, tilt: 0.22, color: [255, 220, 120], a: 0.22 },
      { y: 0.78, amp: 0.06, thick: 0.04, speed: 0.09, ph: 4.4, tilt: 0.16, color: [255, 255, 255], a: 0.55 },
      { y: 0.18, amp: 0.05, thick: 0.03, speed: 0.07, ph: 5.2, tilt: 0.42, color: [255, 255, 255], a: 0.30 },
    ];
    const resize = () => {
      w = cv.clientWidth; h = cv.clientHeight;
      cv.width = Math.max(1, Math.floor(w * SCALE)); cv.height = Math.max(1, Math.floor(h * SCALE));
      ctx.setTransform(SCALE, 0, 0, SCALE, 0, 0);
    };
    const move = (e: PointerEvent) => { mouse.tx = e.clientX / w - 0.5; mouse.ty = e.clientY / h - 0.5; };
    resize(); window.addEventListener("resize", resize); window.addEventListener("pointermove", move);

    const curve = (r: typeof ribbons[0], t: number, off: number) => {
      const pts: [number, number][] = [];
      for (let i = 0; i <= 24; i++) {
        const x = (i / 24) * 1.3 - 0.15;
        const y = r.y + (x - 0.5) * -r.tilt + Math.sin(x * 3.2 + t * r.speed * 6 + r.ph) * r.amp
                  + Math.sin(x * 7.1 - t * r.speed * 4 + r.ph * 2) * r.amp * 0.25 + off + mouse.y * 0.03;
        pts.push([x * w + mouse.x * 30, y * h]);
      }
      return pts;
    };
    const frame = (now: number) => {
      const t = (now - t0) / 1000 * (calm ? 0.5 : 1);
      mouse.x += (mouse.tx - mouse.x) * 0.04; mouse.y += (mouse.ty - mouse.y) * 0.04;
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = "lighter";
      for (const r of ribbons) {
        const top = curve(r, t, -r.thick / 2), bot = curve(r, t + 0.7, r.thick / 2);
        const g = ctx.createLinearGradient(0, (r.y - r.thick) * h, 0, (r.y + r.thick) * h);
        const [cr, cg, cb] = r.color;
        g.addColorStop(0, `rgba(${cr},${cg},${cb},0)`);
        g.addColorStop(0.5, `rgba(${cr},${cg},${cb},${r.a * (calm ? 0.6 : 1)})`);
        g.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
        ctx.fillStyle = g; ctx.beginPath();
        top.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        for (let i = bot.length - 1; i >= 0; i--) ctx.lineTo(bot[i][0], bot[i][1]);
        ctx.closePath(); ctx.fill();
        // crisp highlight edge
        ctx.strokeStyle = `rgba(255,255,255,${0.22 * (calm ? 0.6 : 1)})`; ctx.lineWidth = 1.2; ctx.beginPath();
        top.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
      }
      ctx.globalCompositeOperation = "source-over";
      raf = requestAnimationFrame(frame);
    };
    const vis = () => { cancelAnimationFrame(raf); if (!document.hidden) raf = requestAnimationFrame(frame); };
    raf = requestAnimationFrame(frame);
    document.addEventListener("visibilitychange", vis);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", resize);
                   window.removeEventListener("pointermove", move); document.removeEventListener("visibilitychange", vis); };
  }, [calm]);
  return <canvas ref={ref} aria-hidden className="pointer-events-none fixed inset-0 h-full w-full" />;
}
