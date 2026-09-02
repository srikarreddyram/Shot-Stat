import { useEffect, useRef, useState } from "react";

/**
 * Scroll-synced video background for the splash page.
 *
 * One clip per section, cross-faded on section change. Clips are only given a
 * `src` when they are the active section or adjacent to it — all six eagerly
 * loaded would be ~9.5MB on first paint. Posters stand in until a clip has
 * enough data to show a frame, so a section is never blank.
 *
 * Footage: Mixkit (Mixkit License — free for commercial use, no attribution
 * required). See public/clips/CREDITS.md.
 */

type Clip = {
  src: string;
  poster: string;
  /** object-position focal point */ focus?: string;
};

const CLIPS: Clip[] = [
  // Chosen for texture over gameplay: silhouette, aerial and tight detail read
  // as deliberate art direction, where wide rec-gym footage reads as stock.
  { src: "/clips/00-handle.mp4", poster: "/clips/00-handle.jpg", focus: "50% 55%" },
  { src: "/clips/01-contest.mp4", poster: "/clips/01-contest.jpg", focus: "50% 45%" },
  { src: "/clips/02-shot.mp4", poster: "/clips/02-shot.jpg", focus: "55% 45%" },
  { src: "/clips/03-drive.mp4", poster: "/clips/03-drive.jpg", focus: "50% 50%" },
  { src: "/clips/04-finish.mp4", poster: "/clips/04-finish.jpg", focus: "50% 50%" },
  { src: "/clips/05-net.mp4", poster: "/clips/05-net.jpg", focus: "50% 50%" },
];

/**
 * Bright daylight gym footage against a near-black gold-accented layout reads
 * as a stock-video collage, so every clip is graded down to the palette
 * rather than dropped in raw.
 */
const GRADE = "saturate(0.38) contrast(1.22) brightness(0.42) sepia(0.14)";

export default function VideoBackdrop({ section }: { section: number }) {
  const refs = useRef<(HTMLVideoElement | null)[]>([]);
  // Which clips are allowed to hold a `src` — grows as the reader scrolls.
  const [armed, setArmed] = useState<boolean[]>(() => CLIPS.map((_, i) => i === 0));
  const reduced =
    typeof window !== "undefined" &&
    (window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false);

  useEffect(() => {
    setArmed((prev) => {
      const next = [...prev];
      let changed = false;
      // The active clip plus its neighbours, so the next section is warm.
      for (const i of [section - 1, section, section + 1]) {
        if (i >= 0 && i < CLIPS.length && !next[i]) {
          next[i] = true;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [section]);

  useEffect(() => {
    refs.current.forEach((v, i) => {
      if (!v) return;
      if (i === section && !reduced) {
        // play() rejects when autoplay is blocked; the poster stays up.
        void v.play().catch(() => {});
      } else {
        v.pause();
      }
    });
  }, [section, armed, reduced]);

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 0,
        overflow: "hidden",
        background: "#0a0a0f",
      }}
    >
      {CLIPS.map((clip, i) => (
        <video
          key={clip.src}
          ref={(el) => {
            refs.current[i] = el;
          }}
          src={armed[i] ? clip.src : undefined}
          poster={clip.poster}
          muted
          loop
          playsInline
          preload={i === 0 ? "auto" : "metadata"}
          aria-hidden
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "cover",
            objectPosition: clip.focus ?? "50% 50%",
            filter: GRADE,
            opacity: i === section ? 1 : 0,
            transition: "opacity 900ms ease",
          }}
        />
      ))}
      {/* Gold cast, so the footage sits in the same palette as the copy. */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "linear-gradient(180deg, rgba(201,168,76,0.10), rgba(120,70,20,0.16))",
          mixBlendMode: "overlay",
          pointerEvents: "none",
        }}
      />
    </div>
  );
}
