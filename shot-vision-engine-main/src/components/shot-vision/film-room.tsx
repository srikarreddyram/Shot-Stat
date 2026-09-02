import { useState } from "react";

/**
 * Real NBA footage of the players the engine models, via YouTube's embed
 * player.
 *
 * Every clip below was verified to come from the league's own channel
 * (oEmbed `author_name` == "NBA", youtube.com/@NBA) — not a fan re-upload —
 * so the rights holder published it and enabled embedding. Nothing is
 * downloaded or re-hosted: YouTube serves the video, keeps its branding and
 * gets the view.
 *
 * Cards render as a facade: the poster is a static thumbnail and no request
 * reaches YouTube until someone actually clicks play. That keeps the splash
 * page light and avoids loading third-party player code for visitors who
 * never watch anything.
 */

type Film = {
  player: string;
  /** YouTube id from the official @NBA channel. */
  id: string;
  line: string;
  /** Best/worst zone EP for this player, from player_zone_stats (2025-26). */
  best: [string, number];
  worst: [string, number];
};

const FILMS: Film[] = [
  {
    player: "LUKA DONČIĆ",
    id: "wOMgaCdljbQ",
    line: "38-point triple-double · Jan 20, 2026",
    best: ["RESTRICTED AREA", 1.63],
    worst: ["LEFT CORNER 3", 0.4],
  },
  {
    player: "VICTOR WEMBANYAMA",
    id: "Rf31CLmh45M",
    line: "32 PTS · 8 REB · 6 AST · Finals Game 3",
    best: ["RESTRICTED AREA", 1.5],
    worst: ["MID-RANGE", 0.82],
  },
  {
    player: "CADE CUNNINGHAM",
    id: "kMaqBSTpyko",
    line: "42 PTS · 13 AST at MSG · Feb 19, 2026",
    best: ["RESTRICTED AREA", 1.26],
    worst: ["IN THE PAINT", 0.84],
  },
  {
    player: "JAYSON TATUM",
    id: "FKG8pNICgrA",
    line: "30 PTS · 11 AST · passes Larry Bird",
    best: ["RESTRICTED AREA", 1.23],
    worst: ["MID-RANGE", 0.7],
  },
];

function FilmCard({ film }: { film: Film }) {
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState(false);

  return (
    <div
      style={{
        border: "1px solid rgba(201,168,76,0.18)",
        borderRadius: 4,
        overflow: "hidden",
        background: "#0d0d14",
        transition: "border-color 300ms ease, transform 300ms ease",
        borderColor: hover ? "rgba(201,168,76,0.45)" : "rgba(201,168,76,0.18)",
        transform: hover && !playing ? "translateY(-3px)" : "none",
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <div style={{ position: "relative", aspectRatio: "16 / 9", background: "#000" }}>
        {playing ? (
          <iframe
            // nocookie host: no tracking cookie until playback actually starts.
            src={`https://www.youtube-nocookie.com/embed/${film.id}?autoplay=1&rel=0&modestbranding=1`}
            title={`${film.player} — official NBA highlights`}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            allowFullScreen
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", border: 0 }}
          />
        ) : (
          <button
            onClick={() => setPlaying(true)}
            aria-label={`Play ${film.player} highlights on YouTube`}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              padding: 0,
              border: 0,
              cursor: "pointer",
              background: `#000 url(https://i.ytimg.com/vi/${film.id}/hqdefault.jpg) center/cover no-repeat`,
            }}
          >
            <span
              style={{
                position: "absolute",
                inset: 0,
                background: hover
                  ? "linear-gradient(180deg, rgba(10,10,15,0.25), rgba(10,10,15,0.72))"
                  : "linear-gradient(180deg, rgba(10,10,15,0.45), rgba(10,10,15,0.82))",
                transition: "background 300ms ease",
              }}
            />
            <span
              style={{
                position: "absolute",
                left: "50%",
                top: "50%",
                transform: "translate(-50%,-50%)",
                width: 54,
                height: 54,
                borderRadius: "50%",
                background: hover ? "#C9A84C" : "rgba(201,168,76,0.85)",
                display: "grid",
                placeItems: "center",
                transition: "background 300ms ease",
              }}
            >
              <span
                style={{
                  width: 0,
                  height: 0,
                  marginLeft: 4,
                  borderLeft: "16px solid #000",
                  borderTop: "10px solid transparent",
                  borderBottom: "10px solid transparent",
                }}
              />
            </span>
          </button>
        )}
      </div>

      <div style={{ padding: "14px 16px 16px" }}>
        <div
          style={{
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: 22,
            color: "#F0F0F0",
            letterSpacing: "0.03em",
            lineHeight: 1.1,
          }}
        >
          {film.player}
        </div>
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 9,
            color: "#6b7a92",
            letterSpacing: "0.14em",
            marginTop: 5,
          }}
        >
          {film.line}
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: 14,
            paddingTop: 12,
            borderTop: "1px solid rgba(255,255,255,0.07)",
          }}
        >
          {(
            [
              ["BEST", film.best, "#C9A84C"],
              ["WORST", film.worst, "#DC2626"],
            ] as const
          ).map(([label, [zone, ep], tone]) => (
            <div key={label}>
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 8,
                  color: "#6b7a92",
                  letterSpacing: "0.2em",
                }}
              >
                {label} ZONE
              </div>
              <div
                style={{
                  fontFamily: "'Bebas Neue', sans-serif",
                  fontSize: 24,
                  color: tone,
                  lineHeight: 1.1,
                  marginTop: 2,
                }}
              >
                {ep.toFixed(2)} <span style={{ fontSize: 11, color: "#6b7a92" }}>EP</span>
              </div>
              <div
                style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 8,
                  color: "#6b7a92",
                  letterSpacing: "0.1em",
                }}
              >
                {zone}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function FilmRoom() {
  return (
    <section style={{ background: "#0a0a0f", padding: "110px 6vw 130px" }}>
      <div style={{ maxWidth: 1240, margin: "0 auto" }}>
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            color: "#C9A84C",
            letterSpacing: "0.5em",
          }}
        >
          — FILM ROOM
        </div>
        <div
          style={{
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: 82,
            color: "#F0F0F0",
            lineHeight: 0.95,
            marginTop: 18,
          }}
        >
          Watch the shot.
          <br />
          Then read the number.
        </div>
        <p
          style={{
            fontFamily: "'Inter', sans-serif",
            fontSize: 16,
            color: "#9aa7bd",
            maxWidth: 560,
            lineHeight: 1.55,
            marginTop: 22,
          }}
        >
          Four players the engine models, and what its zone data says about each of them. Expected
          points is field-goal percentage times shot value, from the same table the matchup engine
          reads.
        </p>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(270px, 1fr))",
            gap: 22,
            marginTop: 46,
          }}
        >
          {FILMS.map((f) => (
            <FilmCard key={f.id} film={f} />
          ))}
        </div>

        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 9,
            color: "#4a5568",
            letterSpacing: "0.16em",
            marginTop: 34,
            lineHeight: 1.7,
          }}
        >
          HIGHLIGHTS PUBLISHED BY THE NBA ON YOUTUBE (youtube.com/@NBA) AND PLAYED HERE THROUGH
          YOUTUBE&apos;S EMBED PLAYER.
          <br />
          ZONE FIGURES: PLAYER_ZONE_STATS, 2025-26 REGULAR SEASON.
        </div>
      </div>
    </section>
  );
}
