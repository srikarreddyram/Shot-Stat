import { useState } from "react";

/**
 * A tappable row that expands an explanation in place.
 *
 * The matchup panel is dense with paired numbers ("79% vs 47%", "8\" DEFENDER")
 * that are meaningless unless you already know which side is which and what
 * the pairing is measuring. Rather than shrink the data down or bolt on
 * tooltips that never appear on touch, each block can be opened for a
 * plain-English reading of the exact numbers on screen.
 */
export function Disclosure({ label, children, tone = "#C9A84C", dense = false }: { label: React.ReactNode; children: React.ReactNode; tone?: string; dense?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          width: "100%",
          background: "transparent",
          border: "none",
          padding: dense ? 0 : "2px 0",
          cursor: "pointer",
          textAlign: "left",
          color: "inherit",
          font: "inherit",
        }}
      >
        {label}
        <span
          aria-hidden
          style={{
            marginLeft: "auto",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 9,
            color: tone,
            border: `1px solid ${tone}55`,
            borderRadius: 2,
            padding: "1px 5px",
            flexShrink: 0,
          }}
        >
          {open ? "HIDE" : "WHAT?"}
        </span>
      </button>
      {open && (
        <div
          style={{
            marginTop: 8,
            padding: "10px 12px",
            background: "#0d0d14",
            borderLeft: `2px solid ${tone}`,
            borderRadius: 2,
            fontFamily: "'Inter', sans-serif",
            fontSize: 12.5,
            lineHeight: 1.6,
            color: "#9aa7bd",
          }}
        >
          {children}
        </div>
      )}
    </div>
  );
}
