import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import SplashPage from "@/components/shot-vision/SplashPage";
import EnginePage from "@/components/shot-vision/EnginePage";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "SHOT VISION — NBA Intelligence" },
      { name: "description", content: "Shot Engine and Stat Engine: where to shoot against any defender, every stat for every player, and the shot archetypes the league takes. Built on 2.2M shots across 10 seasons." },
      { property: "og:title", content: "SHOT VISION — NBA Intelligence" },
      { property: "og:description", content: "Where to shoot against any defender, and what every player actually does." },
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
