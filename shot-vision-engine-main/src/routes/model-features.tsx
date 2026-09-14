import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { NBAEdgeBezel, NBADefaultWash, LeagueBrandMark } from "../components/stat-engine/ui";

// Model Features — every feature the live shot-quality model reasons with,
// published as data rather than buried in src/features/spec.py. This
// answers "what does the MODEL use and how much" — model transparency, not
// player/team stats browsing (that's /stats). Self-contained and additive,
// same as /archetypes: it does not import from or modify EnginePage.tsx,
// which is the core recommendation flow.
//
// The number that matters on this page is `gain_share` — XGBoost's gain
// importance normalised across the whole model. A feature can exist in the
// matrix and do nothing; this is what separates the two, and it's why the
// groups are ordered by it rather than alphabetically or by feature count.
//
// This page used to live at /stats. It moved here once "Stat Engine" was
// clarified to mean a player/team stats browser, not model introspection —
// see src/routes/stats.tsx and src/inference/stat_engine.py.

export const Route = createFileRoute("/model-features")({
  head: () => ({
    meta: [
      { title: "SHOT VISION — Model Features" },
      { name: "description", content: "Every engineered feature behind the shot-quality model, grouped and ranked by how much the model actually leans on it." },
    ],
  }),
  component: ModelFeaturesPage,
});

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

type Feature = {
  name: string;
  label: string;
  high: string | null;
  low: string | null;
  gain_share: number;
  direction: string | null;
};

type Group = {
  group: string;
  blurb: string | null;
  n_features: number;
  gain_share: number;
  features: Feature[];
};

type FeatureResponse = {
  model_name: string;
  n_features: number;
  metrics: Record<string, number>;
  groups: Group[];
};

const GOLD = "#C9A84C";

// Groups whose names read as jargon on their own get a plainer heading.
const GROUP_TITLES: Record<string, string> = {
  shooter_skill: "Shooter skill",
  shooter_physical: "Shooter physicals",
  creation: "Creation skill",
  defender: "The defender",
  defender_physical: "Defender physicals",
  opponent_defence: "Opponent team defence",
  interaction: "Matchup interaction",
  spatial: "Court location",
  spatial_basis: "Spatial basis",
  context: "Game state",
  shot_context: "Play-by-play context",
  contest: "Contest level",
  team_creation: "Teammates on the floor",
  help_defense: "Help defenders on the floor",
  shot_mechanic: "Shot mechanic",
  finish: "Finish type",
  zone_indicator: "Zone",
  position_indicator: "Position",
  possession_origin: "Possession origin",
  other: "Other",
};

function ModelFeaturesPage() {
  const [data, setData] = useState<FeatureResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openGroup, setOpenGroup] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/stats/features`)
      .then((r) => {
        if (!r.ok) throw new Error(`engine returned ${r.status}`);
        return r.json();
      })
      .then((d: FeatureResponse) => {
        if (cancelled) return;
        setData(d);
        // Open the heaviest group by default so the page shows real content
        // on arrival rather than a wall of collapsed headers.
        setOpenGroup(d.groups[0]?.group ?? null);
      })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : "Could not load"); });
    return () => { cancelled = true; };
  }, []);

  const maxGroupShare = data ? Math.max(...data.groups.map((g) => g.gain_share), 1e-9) : 1;

  return (
    <div style={{ minHeight: "100vh", background: "#0B0B0F", color: "#F0F0F0", fontFamily: "'Inter', sans-serif", padding: "32px 24px" }}>
      <NBAEdgeBezel />
      <NBADefaultWash />
      <div style={{ maxWidth: 900, margin: "0 auto", position: "relative" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <LeagueBrandMark height={22} />
            <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 34, letterSpacing: 1, color: GOLD }}>
              MODEL FEATURES
            </div>
          </div>
          <a href="/#engine" style={{ color: "#64748b", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: "0.3em", textDecoration: "none" }}>
            ← ENGINE
          </a>
        </div>
        <div style={{ fontSize: 13, color: "rgba(240,240,240,0.55)", marginTop: 4, marginBottom: 6 }}>
          {data
            ? `${data.n_features} engineered features behind ${data.model_name}, grouped and ranked by how much the trained model actually leans on each one.`
            : "Loading..."}
        </div>
        <div style={{ fontSize: 12, color: "rgba(240,240,240,0.38)", marginBottom: 24, lineHeight: 1.55 }}>
          Share is XGBoost gain importance normalised across the whole model — a feature can sit in the
          matrix and do nothing, and this is the number that tells the two apart.
        </div>

        {data && Object.keys(data.metrics).length > 0 && (
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap", background: "#15151D", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 4, padding: "14px 18px", marginBottom: 24 }}>
            {([
              ["log loss", data.metrics.log_loss, 4],
              ["AUC", data.metrics.auc, 4],
              ["accuracy", data.metrics.accuracy, 4],
              ["calibration err", data.metrics.ece, 4],
              ["zone-rank rho", data.metrics.zone_rank_rho, 4],
            ] as [string, number | undefined, number][]).map(([label, value, dp]) =>
              value == null ? null : (
                <div key={label}>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#64748b", letterSpacing: "0.18em", textTransform: "uppercase" }}>
                    {label}
                  </div>
                  <div style={{ fontWeight: 600, fontSize: 15, marginTop: 3 }}>{value.toFixed(dp)}</div>
                </div>
              )
            )}
          </div>
        )}

        {error && (
          <div style={{ color: "#D9484C", fontSize: 13, marginBottom: 16 }}>
            {error} — is the engine running?
          </div>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {data?.groups.map((g) => {
            const isOpen = openGroup === g.group;
            return (
              <div key={g.group} style={{ background: "#15151D", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 4, overflow: "hidden" }}>
                <button
                  onClick={() => setOpenGroup(isOpen ? null : g.group)}
                  aria-expanded={isOpen}
                  style={{
                    width: "100%", textAlign: "left", background: "transparent", border: "none",
                    padding: "14px 18px", cursor: "pointer", color: "inherit", font: "inherit",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
                    <div style={{ fontSize: 15, fontWeight: 600 }}>
                      {GROUP_TITLES[g.group] ?? g.group}
                      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b", marginLeft: 8 }}>
                        {g.n_features} {g.n_features === 1 ? "feature" : "features"}
                      </span>
                    </div>
                    <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 20, color: GOLD, lineHeight: 1 }}>
                      {(g.gain_share * 100).toFixed(1)}%
                      <span style={{ fontSize: 10, color: GOLD, marginLeft: 6 }}>{isOpen ? "▾" : "▸"}</span>
                    </div>
                  </div>
                  <div style={{ height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2, marginTop: 8 }}>
                    <div style={{ width: `${Math.max(2, (g.gain_share / maxGroupShare) * 100)}%`, height: "100%", background: GOLD, borderRadius: 2, opacity: 0.8 }} />
                  </div>
                  {g.blurb && (
                    <div style={{ fontSize: 12, color: "rgba(240,240,240,0.45)", marginTop: 8, lineHeight: 1.5 }}>
                      {g.blurb}
                    </div>
                  )}
                </button>

                {isOpen && (
                  <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", padding: "10px 18px 16px" }}>
                    {g.features.map((f) => {
                      const maxInGroup = Math.max(...g.features.map((x) => x.gain_share), 1e-9);
                      return (
                        <div key={f.name} style={{ padding: "8px 0", borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline" }}>
                            <div style={{ minWidth: 0 }}>
                              <div style={{ fontSize: 13 }}>{f.label}</div>
                              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b", marginTop: 2 }}>
                                {f.name}
                                {f.direction && (
                                  <span style={{ color: GOLD, marginLeft: 8 }}>constrained: {f.direction}</span>
                                )}
                              </div>
                              {(f.high || f.low) && (
                                <div style={{ fontSize: 11, color: "rgba(240,240,240,0.4)", marginTop: 3 }}>
                                  high = {f.high ?? "—"} · low = {f.low ?? "—"}
                                </div>
                              )}
                            </div>
                            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "rgba(240,240,240,0.7)", whiteSpace: "nowrap" }}>
                              {(f.gain_share * 100).toFixed(2)}%
                            </div>
                          </div>
                          <div style={{ height: 2, background: "rgba(255,255,255,0.05)", borderRadius: 2, marginTop: 6 }}>
                            <div style={{ width: `${Math.max(1, (f.gain_share / maxInGroup) * 100)}%`, height: "100%", background: "rgba(201,168,76,0.55)", borderRadius: 2 }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
