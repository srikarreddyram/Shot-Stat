import { useEffect, useRef, useState } from "react";

interface Props {
  onLaunch: () => void;
}

const SECTIONS = 6;

const KEYFRAMES = `
@keyframes wordDrop { from { opacity:0; transform:translateY(-36px); filter:blur(5px);} to { opacity:1; transform:translateY(0); filter:blur(0);} }
@keyframes fadeUp { from { opacity:0; transform:translateY(14px);} to { opacity:1; transform:translateY(0);} }
@keyframes statPop { from { opacity:0; transform:scale(0.85) translateY(12px);} to { opacity:1; transform:scale(1) translateY(0);} }
@keyframes goldPulse { 0%,100% { box-shadow: 0 0 20px rgba(201,168,76,0.2);} 50% { box-shadow: 0 0 48px rgba(201,168,76,0.5);} }
@keyframes scrollNudge { 0%,100% { opacity:0.3; transform:scaleY(0.8);} 50% { opacity:1; transform:scaleY(1.1);} }
@keyframes barFill { from { width:0;} }
@keyframes ambientPulse { 0%,100% { opacity:0.4;} 50% { opacity:0.9;} }
`;

function WordLine({ text, active, delay = 0, size = 110 }: { text: string; active: boolean; delay?: number; size?: number }) {
  const words = text.split(" ");
  return (
    <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: size, color: "#F0F0F0", lineHeight: 0.9, letterSpacing: "0.01em", display: "flex", flexWrap: "wrap", gap: "0.28em" }}>
      {words.map((w, i) => (
        <span key={`${w}-${i}`} style={{ display: "inline-block", opacity: 0, animation: active ? `wordDrop 0.7s cubic-bezier(0.2,0.7,0.2,1) ${delay + i * 0.09}s forwards` : undefined }}>
          {w}
        </span>
      ))}
    </div>
  );
}

export default function SplashPage({ onLaunch }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [section, setSection] = useState(0);
  const sectionRef = useRef(0);

  // Three.js
  useEffect(() => {
    let raf = 0;
    let disposed = false;
    let renderer: any, scene: any, camera: any;
    let attacker: any, defender: any, ball: any;
    const initial = {
      attacker: { x: -1.2, y: 0.55, z: 0 },
      defender: { x: 2.5, y: 0.55, z: 1.5 },
    };

    (async () => {
      // @ts-ignore - dynamic CDN import
      const THREE: any = await import(/* @vite-ignore */ ("https://unpkg.com/three@0.157.0/build/three.module.js" as string));
      if (disposed || !canvasRef.current) return;

      const canvas = canvasRef.current;
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setClearColor(0x0a0a0f);
      const resize = () => {
        const w = window.innerWidth;
        const h = window.innerHeight;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      };

      scene = new THREE.Scene();
      // Denser than the original 0.035 — at that density the floor plane's
      // far edge stayed visible against the clear color, producing a hard
      // horizon line across the whole viewport instead of fading to black.
      scene.fog = new THREE.FogExp2(0x0a0a0f, 0.09);
      camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 100);
      camera.position.set(0, 2.8, 7);
      camera.lookAt(0, 0.5, 0);

      // Floor
      const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(28, 15),
        new THREE.MeshStandardMaterial({ color: 0x5c3317, roughness: 0.95, metalness: 0 })
      );
      floor.rotation.x = -Math.PI / 2;
      floor.position.y = -0.01;
      scene.add(floor);
      const grid = new THREE.GridHelper(28, 28, 0x3d2209, 0x3d2209);
      (grid.material as any).opacity = 0.35;
      (grid.material as any).transparent = true;
      scene.add(grid);

      const makeFigure = (color: number, emissive: number, intensity: number) => {
        const g = new THREE.Group();
        const mat = new THREE.MeshStandardMaterial({ color, emissive, emissiveIntensity: intensity, roughness: 0.5 });
        const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.32, 1.1, 24), mat);
        torso.position.y = 0;
        const head = new THREE.Mesh(new THREE.SphereGeometry(0.25, 24, 24), mat);
        head.position.y = 0.75;
        g.add(torso);
        g.add(head);
        return g;
      };

      attacker = makeFigure(0xc9a84c, 0xc9a84c, 0.4);
      attacker.position.set(initial.attacker.x, initial.attacker.y, initial.attacker.z);
      scene.add(attacker);

      defender = makeFigure(0xdc2626, 0xdc2626, 0.25);
      defender.position.set(initial.defender.x, initial.defender.y, initial.defender.z);
      scene.add(defender);

      // Ball
      const ballGeo = new THREE.SphereGeometry(0.22, 32, 32);
      const ballMat = new THREE.MeshStandardMaterial({ color: 0xc85a00, roughness: 0.8 });
      ball = new THREE.Mesh(ballGeo, ballMat);
      // Seam lines
      const seamMat = new THREE.LineBasicMaterial({ color: 0x1a0a00 });
      const mkCircle = (rotEuler: [number, number, number]) => {
        const pts: any[] = [];
        const segs = 64;
        for (let i = 0; i <= segs; i++) {
          const t = (i / segs) * Math.PI * 2;
          pts.push(new THREE.Vector3(Math.cos(t) * 0.221, Math.sin(t) * 0.221, 0));
        }
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const line = new THREE.LineSegments(geo, seamMat);
        line.rotation.set(rotEuler[0], rotEuler[1], rotEuler[2]);
        ball.add(line);
      };
      mkCircle([0, 0, 0]);
      mkCircle([Math.PI / 2, 0, 0]);
      mkCircle([0, Math.PI / 2, 0]);
      scene.add(ball);

      // Lights
      scene.add(new THREE.AmbientLight(0xffffff, 0.2));
      const key = new THREE.DirectionalLight(0xc9a84c, 1.1);
      key.position.set(4, 6, 3);
      scene.add(key);
      const fill = new THREE.DirectionalLight(0x4466ff, 0.35);
      fill.position.set(-5, 4, -2);
      scene.add(fill);
      const spot = new THREE.SpotLight(0xffffff, 0.6);
      spot.position.set(0, 10, 0);
      spot.target.position.set(0, 0, 0);
      scene.add(spot);
      scene.add(spot.target);
      [[8, 7, 5], [-8, 7, 5], [8, 7, -5], [-8, 7, -5]].forEach(([x, y, z]) => {
        const p = new THREE.PointLight(0xfff5cc, 0.12);
        p.position.set(x, y, z);
        scene.add(p);
      });

      resize();
      window.addEventListener("resize", resize);

      const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
      const smooth = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t));

      const start = performance.now();
      const tick = () => {
        const now = performance.now();
        const t = (now - start) / 1000;
        const s = sectionRef.current;

        // Attacker: idle breathing at 0, move+jump in 1, hold jump 2+
        const forwardProgress = smooth(s >= 1 ? 1 : 0);
        const jumpProgress = smooth(s >= 1 ? Math.min(1, (s - 0.5) * 1) : 0);
        const targetAx = initial.attacker.x;
        const targetAz = initial.attacker.z - 0.8 * forwardProgress;
        const targetAy = initial.attacker.y + 1.8 * jumpProgress;
        attacker.position.x = lerp(attacker.position.x, targetAx, 0.06);
        attacker.position.z = lerp(attacker.position.z, targetAz, 0.06);
        attacker.position.y = lerp(attacker.position.y, targetAy, 0.08);
        if (s === 0) {
          const breath = 1 + Math.sin(t * 3.1) * 0.015;
          attacker.scale.setScalar(breath);
        } else {
          attacker.scale.setScalar(1);
        }

        // Defender closeout
        const closeout = smooth(s >= 1 ? 1 : 0);
        defender.position.x = lerp(defender.position.x, initial.defender.x + (0.9 - initial.defender.x) * closeout, 0.06);
        defender.position.z = lerp(defender.position.z, initial.defender.z + (0.2 - initial.defender.z) * closeout, 0.06);

        // Ball — held at hip height (attacker.y + ~0.35) and offset to the
        // side, not centered on the body: at head height (attacker.y +
        // 0.75, where the head mesh sits) and dead-center on x/z, it read
        // as a basketball stuck on top of the figure's head instead of a
        // held ball. Rises toward shooting-pocket height on the jump.
        const bx = attacker.position.x + 0.34;
        const by = attacker.position.y + 0.1 + (s >= 1 ? 0.75 * jumpProgress : 0);
        const bz = attacker.position.z + 0.2;
        ball.position.x = lerp(ball.position.x, bx, 0.12);
        ball.position.y = lerp(ball.position.y, by, 0.12);
        ball.position.z = lerp(ball.position.z, bz, 0.12);
        if (s >= 2) ball.rotation.y += 0.008;

        // Camera sway
        camera.position.x = Math.sin(now * 0.0004) * 0.4;
        camera.position.z = 7 - s * 0.3;
        camera.lookAt(0, 0.5, 0);

        renderer.render(scene, camera);
        raf = requestAnimationFrame(tick);
      };
      tick();

      return () => window.removeEventListener("resize", resize);
    })();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      renderer?.dispose?.();
    };
  }, []);

  // Scroll -> section
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onScroll = () => {
      const rect = el.getBoundingClientRect();
      const scrolled = Math.min(1, Math.max(0, -rect.top / (rect.height - window.innerHeight)));
      const idx = Math.min(SECTIONS - 1, Math.floor(scrolled * SECTIONS));
      sectionRef.current = idx;
      setSection(idx);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const canvasOpacity = section <= 2 ? 1 : Math.max(0.1, 1 - (section - 2) * 0.3);

  return (
    <div ref={wrapRef} style={{ height: "600vh", background: "#0a0a0f", position: "relative" }}>
      <style>{KEYFRAMES}</style>
      <div style={{ position: "sticky", top: 0, height: "100vh", overflow: "hidden", background: "#0a0a0f" }}>
        <canvas
          ref={canvasRef}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", zIndex: 0, opacity: canvasOpacity, transition: "opacity 0.6s ease" }}
        />
        {/* Softens the 3D floor's horizon line — GridHelper lines don't
            respect scene.fog, so they stay crisp against the fogged floor
            color and read as a hard seam across the viewport. A blurred
            band over just that strip blends it without touching the
            WebGL scene itself. */}
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: "31%",
            height: "10%",
            zIndex: 1,
            pointerEvents: "none",
            backdropFilter: "blur(18px)",
            WebkitBackdropFilter: "blur(18px)",
            maskImage: "linear-gradient(to bottom, transparent 0%, black 35%, black 65%, transparent 100%)",
            WebkitMaskImage: "linear-gradient(to bottom, transparent 0%, black 35%, black 65%, transparent 100%)",
            opacity: canvasOpacity,
            transition: "opacity 0.6s ease",
          }}
        />
        {/* Nav dots */}
        <div style={{ position: "absolute", right: 32, top: "50%", transform: "translateY(-50%)", zIndex: 20, display: "flex", flexDirection: "column", gap: 12 }}>
          {Array.from({ length: SECTIONS }).map((_, i) => (
            <div
              key={i}
              onClick={() => {
                const el = wrapRef.current;
                if (!el) return;
                const total = el.offsetHeight - window.innerHeight;
                window.scrollTo({ top: el.offsetTop + (total * i) / SECTIONS + 10, behavior: "smooth" });
              }}
              style={{
                width: section === i ? 3 : 2,
                height: section === i ? 28 : 8,
                background: section === i ? "#C9A84C" : "rgba(201,168,76,0.2)",
                borderRadius: 2,
                cursor: "pointer",
                transition: "all 400ms cubic-bezier(0.34,1.56,0.64,1)",
              }}
            />
          ))}
        </div>

        {/* SECTION 0 */}
        <SectionOverlay active={section === 0} align="center">
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
            <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 200, color: "rgba(201,168,76,0.018)", letterSpacing: "0.02em" }}>VISION</div>
          </div>
          <div style={{ position: "relative", maxWidth: 700, margin: "0 auto", padding: "0 24px", textAlign: "center" }}>
            <Tag>SHOT VISION</Tag>
            <div style={{ marginTop: 24 }}>
              <WordLine text="Every bucket" active={section === 0} />
              <WordLine text="starts here." active={section === 0} delay={0.18} />
            </div>
            <p style={{ marginTop: 32, opacity: 0, animation: section === 0 ? "fadeUp 0.7s ease 0.9s forwards" : undefined, fontFamily: "'Inter', sans-serif", fontWeight: 400, fontSize: 17, color: "#64748b", maxWidth: 460, marginLeft: "auto", marginRight: "auto", lineHeight: 1.55 }}>
              2 seconds. One defender. One decision. SHOT VISION already knows the answer.
            </p>
          </div>
          <div style={{ position: "absolute", bottom: 40, left: "50%", transform: "translateX(-50%)", textAlign: "center" }}>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#C9A84C", letterSpacing: "0.4em" }}>SCROLL</div>
            <div style={{ width: 1, height: 40, background: "#C9A84C", margin: "10px auto 0", animation: "scrollNudge 1.6s ease infinite", transformOrigin: "top" }} />
          </div>
        </SectionOverlay>

        {/* SECTION 1 */}
        <SectionOverlay active={section === 1} align="left">
          <div style={{ paddingLeft: "8vw", maxWidth: 620 }}>
            <Tag>— THE MOMENT</Tag>
            <div style={{ marginTop: 20 }}>
              <WordLine text="The closeout" active={section === 1} size={90} />
              <WordLine text="is coming." active={section === 1} delay={0.2} size={90} />
            </div>
            <p style={{ marginTop: 28, opacity: 0, animation: section === 1 ? "fadeUp 0.7s ease 0.9s forwards" : undefined, fontFamily: "'Inter', sans-serif", fontSize: 16, color: "#64748b", maxWidth: 400, lineHeight: 1.55 }}>
              A 6'9" wing at full sprint. 80 inches of wingspan. You have 0.4 seconds. SHOT VISION already has the answer.
            </p>
            <div style={{ marginTop: 40, opacity: 0, animation: section === 1 ? "statPop 0.6s cubic-bezier(0.34,1.56,0.64,1) 1.3s forwards" : undefined }}>
              <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 72, color: "#C9A84C", lineHeight: 1 }}>0.4</div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b", letterSpacing: "0.4em", marginTop: 6 }}>SECONDS TO DECIDE</div>
            </div>
          </div>
        </SectionOverlay>

        {/* SECTION 2 */}
        <SectionOverlay active={section === 2} align="right">
          <div style={{ paddingRight: "8vw", maxWidth: 560, marginLeft: "auto", textAlign: "right" }}>
            <Tag>— SHOT QUALITY</Tag>
            <div style={{ marginTop: 20 }}>
              <WordLine text="Not all shots" active={section === 2} size={90} />
              <WordLine text="are equal." active={section === 2} delay={0.18} size={90} />
            </div>
            <div style={{ marginTop: 36, opacity: 0, animation: section === 2 ? "fadeUp 0.7s ease 0.9s forwards" : undefined }}>
              <ComparisonRow label="MID-RANGE" pct={52} color="#DC2626" ep="0.72 EP" active={section === 2} delay={1.0} />
              <div style={{ height: 14 }} />
              <ComparisonRow label="RESTRICTED AREA" pct={100} color="#C9A84C" epColor="#16A34A" ep="1.36 EP" active={section === 2} delay={1.2} />
              <div style={{ marginTop: 20, opacity: 0, animation: section === 2 ? "fadeUp 0.6s ease 2s forwards" : undefined, fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#C9A84C" }}>
                +0.64 EP with SHOT VISION
              </div>
            </div>
          </div>
        </SectionOverlay>

        {/* SECTION 3 */}
        <SectionOverlay active={section === 3} align="left">
          <div style={{ paddingLeft: "8vw", maxWidth: 900 }}>
            <Tag>— THE DATASET</Tag>
            <div style={{ marginTop: 20 }}>
              <WordLine text="847,000 shots." active={section === 3} size={90} />
              <WordLine text="12 seasons." active={section === 3} delay={0.2} size={90} />
            </div>
            <div style={{ marginTop: 44, display: "flex", gap: 64, flexWrap: "wrap" }}>
              <StatCallout value="12" label="SEASONS ANALYZED" active={section === 3} delay={1.0} />
              <StatCallout value="847K" label="SHOTS PROCESSED" active={section === 3} delay={1.2} />
              <StatCallout value="<200" label="MS INFERENCE" active={section === 3} delay={1.4} />
            </div>
          </div>
        </SectionOverlay>

        {/* SECTION 4 */}
        <SectionOverlay active={section === 4} align="right">
          <div style={{ paddingRight: "8vw", maxWidth: 620, marginLeft: "auto", textAlign: "right" }}>
            <Tag>— THE INTELLIGENCE</Tag>
            <div style={{ marginTop: 20 }}>
              <WordLine text="XGBoost." active={section === 4} size={90} />
              <WordLine text="Trained on" active={section === 4} delay={0.15} size={90} />
              <WordLine text="mismatches." active={section === 4} delay={0.3} size={90} />
            </div>
            <p style={{ marginTop: 32, marginLeft: "auto", opacity: 0, animation: section === 4 ? "fadeUp 0.7s ease 1.1s forwards" : undefined, fontFamily: "'Inter', sans-serif", fontSize: 16, color: "#64748b", maxWidth: 440, lineHeight: 1.55 }}>
              Height differential. Wingspan. Contest rate. Zone tendencies. Game state. All of it in under 200ms.
            </p>
          </div>
        </SectionOverlay>

        {/* SECTION 5 - CTA */}
        <SectionOverlay active={section === 5} align="center">
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
            <div style={{ width: 600, height: 600, borderRadius: "50%", background: "radial-gradient(circle, rgba(201,168,76,0.08) 0%, rgba(201,168,76,0) 70%)", animation: "ambientPulse 5s ease infinite" }} />
          </div>
          <div style={{ position: "relative", textAlign: "center", padding: "0 24px", maxWidth: 1000, margin: "0 auto" }}>
            <Tag>SHOT VISION — NBA SHOT QUALITY ENGINE</Tag>
            <div style={{ marginTop: 28, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <WordLine text="Know before" active={section === 5} size={130} />
              <WordLine text="you shoot." active={section === 5} delay={0.22} size={130} />
            </div>
            <div style={{ width: 48, height: 1, background: "#C9A84C", margin: "36px auto 0", opacity: 0, animation: section === 5 ? "fadeUp 0.6s ease 1s forwards" : undefined }} />
            <p style={{ marginTop: 20, opacity: 0, animation: section === 5 ? "fadeUp 0.7s ease 1.1s forwards" : undefined, fontFamily: "'Inter', sans-serif", fontSize: 17, color: "#64748b" }}>
              Input the matchup. Run the engine.
            </p>
            <button
              onClick={onLaunch}
              style={{
                marginTop: 40,
                opacity: 0,
                animation: section === 5 ? "statPop 0.6s cubic-bezier(0.34,1.56,0.64,1) 1.4s forwards" : undefined,
                background: "#C9A84C",
                color: "#000",
                border: "none",
                fontFamily: "'Bebas Neue', sans-serif",
                fontSize: 22,
                letterSpacing: "0.05em",
                padding: "18px 52px",
                borderRadius: 3,
                boxShadow: "0 0 40px rgba(201,168,76,0.25)",
                cursor: "pointer",
                transition: "all 300ms ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.boxShadow = "0 0 64px rgba(201,168,76,0.45)";
                e.currentTarget.style.transform = "scale(1.03)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.boxShadow = "0 0 40px rgba(201,168,76,0.25)";
                e.currentTarget.style.transform = "scale(1)";
              }}
            >
              RUN SHOT VISION →
            </button>
            <div style={{ marginTop: 40, opacity: 0, animation: section === 5 ? "fadeUp 0.6s ease 1.7s forwards" : undefined, fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#2a2a3a", letterSpacing: "0.3em" }}>
              SHOT VISION · XGBOOST · 12 SEASONS · &lt;200MS INFERENCE
            </div>
          </div>
        </SectionOverlay>
      </div>
    </div>
  );
}

function SectionOverlay({ children, active, align }: { children: React.ReactNode; active: boolean; align: "left" | "right" | "center" }) {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 10,
        display: "flex",
        alignItems: "center",
        justifyContent: align === "left" ? "flex-start" : align === "right" ? "flex-end" : "center",
        opacity: active ? 1 : 0,
        pointerEvents: active ? "auto" : "none",
        transition: "opacity 0.55s ease",
      }}
    >
      <div style={{ width: "100%" }}>{children}</div>
    </div>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#C9A84C", letterSpacing: "0.5em" }}>{children}</div>;
}

function ComparisonRow({ label, pct, color, epColor, ep, active, delay }: { label: string; pct: number; color: string; epColor?: string; ep: string; active: boolean; delay: number }) {
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 6 }}>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color, letterSpacing: "0.2em" }}>{label}</div>
        <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 32, color: epColor || color, lineHeight: 1 }}>{ep}</div>
      </div>
      <div style={{ height: 6, background: "#1a1a2a", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ height: "100%", width: active ? `${pct}%` : 0, background: color, transition: `width 700ms ease ${delay}s` }} />
      </div>
    </div>
  );
}

function StatCallout({ value, label, active, delay }: { value: string; label: string; active: boolean; delay: number }) {
  return (
    <div style={{ opacity: 0, animation: active ? `statPop 0.6s cubic-bezier(0.34,1.56,0.64,1) ${delay}s forwards` : undefined }}>
      <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 80, color: "#C9A84C", lineHeight: 1 }}>{value}</div>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b", letterSpacing: "0.3em", marginTop: 4 }}>{label}</div>
    </div>
  );
}