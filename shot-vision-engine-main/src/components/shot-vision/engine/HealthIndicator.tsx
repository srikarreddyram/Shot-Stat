export function HealthIndicator({ status }: { status: "checking" | "online" | "offline" }) {
  const color = status === "online" ? "#16A34A" : status === "offline" ? "#DC2626" : "#64748b";
  const label = status === "online" ? "ENGINE ONLINE" : status === "offline" ? "ENGINE OFFLINE" : "CONNECTING…";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
      <div style={{ width: 6, height: 6, borderRadius: "50%", background: color, boxShadow: status === "online" ? `0 0 6px ${color}` : "none", flexShrink: 0 }} />
      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: "#64748b", letterSpacing: "0.25em" }}>{label}</span>
    </div>
  );
}
