import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import SplashPage from "@/components/shot-vision/SplashPage";
import EnginePage from "@/components/shot-vision/EnginePage";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "SHOT VISION — NBA Shot Quality & Matchup Engine" },
      { name: "description", content: "Real-time NBA shot quality and matchup recommendations. XGBoost trained on 847,000 shots across 12 seasons. Know before you shoot." },
      { property: "og:title", content: "SHOT VISION — NBA Shot Quality Engine" },
      { property: "og:description", content: "Input the matchup. Run the engine. Optimal zones, expected points, and make probability in under 200ms." },
    ],
  }),
  component: Index,
});

function Index() {
  const [page, setPage] = useState<"splash" | "engine">(
    typeof window !== "undefined" && window.location.hash === "#engine" ? "engine" : "splash"
  );
  const [visible, setVisible] = useState(true);

  const transitionTo = (next: "splash" | "engine") => {
    setVisible(false);
    setTimeout(() => {
      setPage(next);
      if (typeof window !== "undefined") {
        if (next === "engine") {
          window.history.pushState(null, "", "#engine");
        } else {
          window.history.pushState(null, "", window.location.pathname + window.location.search);
        }
        window.scrollTo(0, 0);
      }
      requestAnimationFrame(() => setVisible(true));
    }, 350);
  };

  useEffect(() => {
    document.body.style.background = "#0a0a0f";
  }, []);

  return (
    <div style={{ opacity: visible ? 1 : 0, transition: `opacity ${visible ? 450 : 350}ms ease`, background: "#0a0a0f", minHeight: "100vh" }}>
      {page === "splash" ? (
        <SplashPage onLaunch={() => transitionTo("engine")} />
      ) : (
        <EnginePage onBack={() => transitionTo("splash")} />
      )}
    </div>
  );
}
