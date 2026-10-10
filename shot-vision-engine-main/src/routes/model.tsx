import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { ArchetypesPanel } from "../components/model/ArchetypesPanel";
import { FeaturesPanel } from "../components/model/FeaturesPanel";
import {
  AppHeader,
  C,
  DEFAULT_ACCENT,
  F,
  MOTION_CSS,
  NBADefaultWash,
  NBAEdgeBezel,
  Segmented,
  accentVars,
} from "../components/stat-engine/ui";

// The model, in one place: what the shot-quality model leans on (its
// features, ranked by gain) and the kinds of shot the league takes (shot
// archetypes, clustered over those same features). These were two pages —
// /model-features and /archetypes — which now redirect to the tabs here.
// The tab lives in the URL hash so either view can be linked to directly.

type Tab = "features" | "archetypes";

export const Route = createFileRoute("/model")({
  head: () => ({
    meta: [
      { title: "SHOT VISION — Model" },
      {
        name: "description",
        content:
          "Inside the shot-quality model: every feature it reasons with, ranked by how much it leans on each, and the shot archetypes found by clustering 2.2M shots.",
      },
    ],
  }),
  component: ModelPage,
});

const tabFromHash = (): Tab =>
  typeof window !== "undefined" && window.location.hash === "#archetypes"
    ? "archetypes"
    : "features";

function ModelPage() {
  const [tab, setTab] = useState<Tab>("features");

  useEffect(() => {
    const sync = () => setTab(tabFromHash());
    sync();
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);

  const choose = (next: Tab) => {
    setTab(next);
    window.history.replaceState(null, "", `#${next}`);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0B0B0F",
        color: "#F0F0F0",
        fontFamily: F.body,
        ...accentVars(DEFAULT_ACCENT),
      }}
    >
      <style>{MOTION_CSS}</style>
      <NBAEdgeBezel />
      <NBADefaultWash />
      <AppHeader active="model" sub="MODEL · FEATURES & SHOT ARCHETYPES" />
      <main
        style={{ maxWidth: 960, margin: "0 auto", position: "relative", padding: "28px 24px 40px" }}
      >
        <div
          style={{
            fontFamily: F.display,
            fontSize: 38,
            letterSpacing: "0.02em",
            color: C.gold,
            lineHeight: 1,
          }}
        >
          INSIDE THE MODEL
        </div>
        <div
          style={{
            fontSize: 13,
            color: C.dim,
            marginTop: 8,
            marginBottom: 18,
            lineHeight: 1.55,
            maxWidth: 720,
          }}
        >
          Two views of the same engine: which of its features the shot-quality model actually leans
          on, and the kinds of shot the whole league takes, found by clustering those same features.
        </div>
        <div style={{ marginBottom: 22 }}>
          <Segmented
            value={tab}
            onChange={(v) => choose(v as Tab)}
            options={[
              { value: "features", label: "WHAT IT LEARNED" },
              { value: "archetypes", label: "SHOT ARCHETYPES" },
            ]}
          />
        </div>
        <div key={tab} className="se-rise">
          {tab === "features" ? <FeaturesPanel /> : <ArchetypesPanel />}
        </div>
      </main>
    </div>
  );
}
