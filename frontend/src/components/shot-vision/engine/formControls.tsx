import { NBA_RED, NBA_BLUE } from "@/components/stat-engine/ui";

export function MiniLabel({ children }: { children: React.ReactNode }) {
  return <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#64748b", letterSpacing: "0.3em" }}>{children}</div>;
}

export function InputSection({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        {/* A tiny red/blue tick repeated on every section label — the one
            recurring motif that makes the league colours read as a running
            design system rather than a one-off header flourish. */}
        <div style={{ display: "flex", height: 10 }} aria-hidden="true">
          <div style={{ width: 3, background: NBA_RED }} />
          <div style={{ width: 3, background: NBA_BLUE }} />
        </div>
        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#C9A84C", letterSpacing: "0.4em" }}>{label}</div>
      </div>
      {children}
    </div>
  );
}

export function Pill({ active, children, onClick, activeColor = "#C9A84C", activeTextColor = "#000" }: { active: boolean; children: React.ReactNode; onClick: () => void; activeColor?: string; activeTextColor?: string }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "10px 20px",
        borderRadius: 3,
        border: active ? "none" : "1px solid #2a2a3a",
        background: active ? activeColor : "transparent",
        color: active ? activeTextColor : "#64748b",
        fontFamily: active ? "'Bebas Neue', sans-serif" : "'JetBrains Mono', monospace",
        fontSize: active ? 15 : 11,
        letterSpacing: active ? "0.05em" : "0.25em",
        cursor: "pointer",
        transition: "all 200ms",
      }}
    >
      {children}
    </button>
  );
}
