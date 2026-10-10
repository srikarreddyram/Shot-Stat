import { useEffect, useRef, useState } from "react";
import { drawCourtLines, drawHardwood } from "../shot-vision/engine/court/courtDrawing";
import { COURT_AREA_H, COURT_CW, courtToCanvas } from "../shot-vision/engine/court/geometry";

// Shot archetypes: real shots clustered by PCA + K-Means over the engine's
// engineered feature set into kinds of shot like "contested rim driving
// layup" (src/analysis/shot_archetypes.py), drawn on the same hardwood court
// as the engine and the Stat Engine's shot maps (courtDrawing.ts). One of the
// two tabs of the /model page, with the model's features.

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

// A fixed, colourblind-considered categorical palette — up to 8 archetypes,
// each a distinct hue rather than a lightness ramp (this is categorical
// data, not a magnitude, so a sequential ramp would be the wrong tool).
const CLUSTER_COLORS = [
  "#4C9ED9",
  "#E0793C",
  "#5CB85C",
  "#D65C7A",
  "#9B7ED9",
  "#C9A84C",
  "#4CBCA0",
  "#D9484C",
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

export function ArchetypesPanel() {
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [points, setPoints] = useState<Record<string, { label: string; points: CourtPoint[] }>>({});
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [modelVersion, setModelVersion] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Two requests, rendered as each arrives: the cluster list is instant, the
  // court sample can take a second on a cold server, and the list should not
  // wait for it.
  useEffect(() => {
    let cancelled = false;
    const get = (path: string) =>
      fetch(`${API_BASE}${path}`).then((r) => {
        if (!r.ok) throw new Error("archetype endpoints returned an error");
        return r.json();
      });
    const fail = (e: unknown) => {
      if (!cancelled) setError(e instanceof Error ? e.message : "failed to load archetypes");
    };
    get("/archetypes")
      .then((def) => {
        if (cancelled) return;
        setClusters(def.clusters);
        setModelVersion(def.model_version);
      })
      .catch(fail);
    get("/archetypes/court?points_per_cluster=300")
      .then((court) => {
        if (!cancelled) setPoints(court.clusters);
      })
      .catch(fail);
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    drawHardwood(ctx, COURT_AREA_H);
    drawCourtLines(ctx);
    for (const cluster of clusters) {
      if (hidden.has(cluster.cluster_id)) continue;
      const entry = points[String(cluster.cluster_id)];
      if (!entry) continue;
      ctx.fillStyle = CLUSTER_COLORS[cluster.cluster_id % CLUSTER_COLORS.length];
      ctx.globalAlpha = 0.7;
      for (const p of entry.points) {
        const { x, y } = courtToCanvas(p.loc_x, p.loc_y);
        if (y < 0 || y > COURT_AREA_H) continue;
        ctx.beginPath();
        ctx.arc(x, y, 2.3, 0, Math.PI * 2);
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
    <div>
      <div
        style={{
          fontSize: 13,
          color: "rgba(240,240,240,0.55)",
          marginTop: 4,
          marginBottom: 24,
          lineHeight: 1.55,
        }}
      >
        {clusters.length > 0
          ? `${clusters.reduce((s, c) => s + c.n_shots, 0).toLocaleString()} real shots, clustered into ${clusters.length} archetypes by PCA + K-Means over the same features the model uses (${modelVersion}). Click a row to show or hide it on the court.`
          : "Loading…"}
      </div>

      {error && (
        <div style={{ color: "#D9484C", fontSize: 13, marginBottom: 16 }}>
          {error} — has `python -m src.analysis.shot_archetypes --fit` been run?
        </div>
      )}

      <div style={{ display: "flex", gap: 28, flexWrap: "wrap", alignItems: "flex-start" }}>
        <canvas
          ref={canvasRef}
          width={COURT_CW}
          height={COURT_AREA_H}
          role="img"
          aria-label="Sample shots from each archetype, on the court"
          style={{
            flex: "1 1 360px",
            maxWidth: 480,
            width: "100%",
            height: "auto",
            borderRadius: 4,
            boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
          }}
        />

        <div
          style={{
            flex: "1 1 260px",
            minWidth: 0,
            display: "flex",
            flexDirection: "column",
            gap: 6,
          }}
        >
          {clusters.map((c) => {
            const isHidden = hidden.has(c.cluster_id);
            return (
              <button
                key={c.cluster_id}
                onClick={() => toggle(c.cluster_id)}
                aria-pressed={!isHidden}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  textAlign: "left",
                  background: "#15151D",
                  border: "1px solid rgba(255,255,255,0.06)",
                  borderRadius: 3,
                  padding: "10px 12px",
                  cursor: "pointer",
                  opacity: isHidden ? 0.4 : 1,
                  color: "inherit",
                  font: "inherit",
                }}
              >
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    background: CLUSTER_COLORS[c.cluster_id % CLUSTER_COLORS.length],
                    flexShrink: 0,
                  }}
                />
                <span style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, textTransform: "capitalize" }}>
                    {c.label}
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      color: "rgba(240,240,240,0.5)",
                      fontFamily: "'JetBrains Mono', monospace",
                    }}
                  >
                    {(c.share_of_shots * 100).toFixed(1)}% of shots · {(c.fg_pct * 100).toFixed(1)}%
                    FG
                  </div>
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
