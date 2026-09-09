import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";

// A new, self-contained page at /archetypes — additive only. It does not
// import from or modify EnginePage.tsx / CourtCanvas, which is the core
// recommendation flow; a small amount of court-geometry duplication below is
// the deliberate trade for zero risk to that file. See
// src/analysis/shot_archetypes.py for what is being visualised: shots
// clustered by PCA + K-Means into archetypes like "contested rim driving
// layup", surfaced here on the existing peach-hardwood court styling.

export const Route = createFileRoute("/archetypes")({
  head: () => ({
    meta: [
      { title: "SHOT VISION — Shot Archetypes" },
      { name: "description", content: "Shot archetypes discovered by PCA + K-Means clustering over the engine's engineered feature set." },
    ],
  }),
  component: ArchetypesPage,
});

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

// ── Court geometry (duplicated from EnginePage.tsx's CourtCanvas on purpose
// — see the file-level note above) ──────────────────────────────────────────
const FT = {
  courtWidth: 50,
  hoopFromBaseline: 5.25,
  rimRadius: 0.75,
  restrictedRadius: 4,
  laneWidth: 16,
  laneDepth: 19,
  ftCircleRadius: 6,
  cornerThreeX: 22,
  arcRadius: 23.75,
  visibleDepth: 38,
};
const CORNER_JOIN_Y = Math.sqrt(FT.arcRadius ** 2 - FT.cornerThreeX ** 2);
const COURT_CW = 460;
const COURT_SCALE = COURT_CW / FT.courtWidth;
const COURT_CH = Math.round(FT.visibleDepth * COURT_SCALE);
const HOOP_X = COURT_CW / 2;
const HOOP_Y = COURT_CH - FT.hoopFromBaseline * COURT_SCALE;
const ft = (feet: number) => feet * COURT_SCALE;

const COURT_SURFACE = "#DCB489";
const COURT_LINE = "rgba(92, 58, 30, 0.5)";
const COURT_LINE_STRONG = "rgba(58, 35, 16, 0.85)";
const COURT_RIM = "#B4491C";

function courtToCanvas(locX: number, locY: number) {
  return { x: HOOP_X + (locX / 10) * COURT_SCALE, y: HOOP_Y - (locY / 10) * COURT_SCALE };
}

// A fixed, colourblind-considered categorical palette — up to 8 archetypes,
// each a distinct hue rather than a lightness ramp (this is categorical
// data, not a magnitude, so a sequential ramp would be the wrong tool).
const CLUSTER_COLORS = [
  "#4C9ED9", "#E0793C", "#5CB85C", "#D65C7A",
  "#9B7ED9", "#C9A84C", "#4CBCA0", "#D9484C",
];

type Cluster = {
  cluster_id: number;
  label: string;
  n_shots: number;
  share_of_shots: number;
  fg_pct: number;
  dominant_zone: string;
  avg_shot_distance: number;
};

type CourtPoint = { loc_x: number; loc_y: number; zone: string };

function drawCourt(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = COURT_SURFACE;
  ctx.fillRect(0, 0, COURT_CW, COURT_CH);

  ctx.strokeStyle = COURT_LINE;
  ctx.lineWidth = 1.5;

  // Paint / lane
  ctx.strokeRect(HOOP_X - ft(FT.laneWidth / 2), HOOP_Y - ft(FT.laneDepth), ft(FT.laneWidth), ft(FT.laneDepth));

  // Free-throw circle
  ctx.beginPath();
  ctx.arc(HOOP_X, HOOP_Y - ft(FT.laneDepth), ft(FT.ftCircleRadius), 0, Math.PI * 2);
  ctx.stroke();

  // Restricted area
  ctx.beginPath();
  ctx.arc(HOOP_X, HOOP_Y, ft(FT.restrictedRadius), Math.PI, 0);
  ctx.stroke();

  // Three-point line: corners + arc
  const cy = HOOP_Y - ft(CORNER_JOIN_Y);
  ctx.beginPath();
  ctx.moveTo(HOOP_X - ft(FT.cornerThreeX), HOOP_Y);
  ctx.lineTo(HOOP_X - ft(FT.cornerThreeX), cy);
  ctx.moveTo(HOOP_X + ft(FT.cornerThreeX), HOOP_Y);
  ctx.lineTo(HOOP_X + ft(FT.cornerThreeX), cy);
  ctx.stroke();

  ctx.beginPath();
  const startAngle = Math.PI + Math.asin(ft(CORNER_JOIN_Y) / ft(FT.arcRadius));
  const endAngle = -Math.asin(ft(CORNER_JOIN_Y) / ft(FT.arcRadius));
  ctx.arc(HOOP_X, HOOP_Y, ft(FT.arcRadius), Math.PI - Math.asin(CORNER_JOIN_Y / FT.arcRadius), Math.asin(CORNER_JOIN_Y / FT.arcRadius), true);
  ctx.stroke();

  // Rim + backboard
  ctx.beginPath();
  ctx.arc(HOOP_X, HOOP_Y, ft(FT.rimRadius), 0, Math.PI * 2);
  ctx.strokeStyle = COURT_RIM;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.strokeStyle = COURT_LINE_STRONG;
  ctx.lineWidth = 1;
  ctx.strokeRect(0, 0, COURT_CW, COURT_CH);
}

function ArchetypesPage() {
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [points, setPoints] = useState<Record<string, { label: string; points: CourtPoint[] }>>({});
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [modelVersion, setModelVersion] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    (async () => {
      try {
        const [defRes, courtRes] = await Promise.all([
          fetch(`${API_BASE}/archetypes`),
          fetch(`${API_BASE}/archetypes/court?points_per_cluster=300`),
        ]);
        if (!defRes.ok || !courtRes.ok) throw new Error("archetype endpoints returned an error");
        const def = await defRes.json();
        const court = await courtRes.json();
        setClusters(def.clusters);
        setModelVersion(def.model_version);
        setPoints(court.clusters);
      } catch (e: any) {
        setError(e.message ?? "failed to load archetypes");
      }
    })();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = COURT_CW;
    canvas.height = COURT_CH;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    drawCourt(ctx);

    for (const cluster of clusters) {
      if (hidden.has(cluster.cluster_id)) continue;
      const entry = points[String(cluster.cluster_id)];
      if (!entry) continue;
      const color = CLUSTER_COLORS[cluster.cluster_id % CLUSTER_COLORS.length];
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.65;
      for (const p of entry.points) {
        const { x, y } = courtToCanvas(p.loc_x, p.loc_y);
        ctx.beginPath();
        ctx.arc(x, y, 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
  }, [clusters, points, hidden]);

  const toggle = (id: number) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0B0B0F", color: "#F0F0F0", fontFamily: "'Inter', sans-serif", padding: "32px 24px" }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 34, letterSpacing: 1, color: "#C9A84C" }}>
          SHOT ARCHETYPES
        </div>
        <div style={{ fontSize: 13, color: "rgba(240,240,240,0.55)", marginTop: 4, marginBottom: 24 }}>
          {clusters.length > 0
            ? `${clusters.reduce((s, c) => s + c.n_shots, 0).toLocaleString()} real shots, clustered into ${clusters.length} archetypes via PCA + K-Means (model: ${modelVersion}). Click a legend row to toggle it on the court.`
            : "Loading..."}
        </div>

        {error && (
          <div style={{ color: "#D9484C", fontSize: 13, marginBottom: 16 }}>
            {error} — has `python -m src.analysis.shot_archetypes --fit` been run?
          </div>
        )}

        <div style={{ display: "flex", gap: 28, flexWrap: "wrap" }}>
          <canvas
            ref={canvasRef}
            style={{ borderRadius: 4, boxShadow: "0 8px 32px rgba(0,0,0,0.4)", maxWidth: "100%", height: "auto" }}
          />

          <div style={{ flex: 1, minWidth: 260, display: "flex", flexDirection: "column", gap: 6 }}>
            {clusters.map((c) => {
              const color = CLUSTER_COLORS[c.cluster_id % CLUSTER_COLORS.length];
              const isHidden = hidden.has(c.cluster_id);
              return (
                <button
                  key={c.cluster_id}
                  onClick={() => toggle(c.cluster_id)}
                  style={{
                    display: "flex", alignItems: "center", gap: 10, textAlign: "left",
                    background: "#15151D", border: "1px solid rgba(255,255,255,0.06)",
                    borderRadius: 3, padding: "10px 12px", cursor: "pointer",
                    opacity: isHidden ? 0.4 : 1,
                  }}
                >
                  <span style={{ width: 10, height: 10, borderRadius: "50%", background: color, flexShrink: 0 }} />
                  <span style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, textTransform: "capitalize" }}>{c.label}</div>
                    <div style={{ fontSize: 11, color: "rgba(240,240,240,0.5)", fontFamily: "'JetBrains Mono', monospace" }}>
                      {(c.share_of_shots * 100).toFixed(1)}% of shots · {(c.fg_pct * 100).toFixed(1)}% FG
                    </div>
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
