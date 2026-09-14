import { useEffect, useState } from "react";
import type { Player } from "@/lib/shot-vision-data";

export function PlayerAvatar({ player, size = 40 }: { player: Player; size?: number }) {
  const [broken, setBroken] = useState(false);
  useEffect(() => { setBroken(false); }, [player.id]);
  const initials = player.name
    .split(" ")
    .filter(Boolean)
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  const baseStyle: React.CSSProperties = {
    width: size,
    height: size,
    borderRadius: "50%",
    flexShrink: 0,
    background: "#1a1a2a",
    border: "1px solid rgba(201,168,76,0.2)",
  };

  if (!player.headshotUrl || broken) {
    return (
      <div style={{ ...baseStyle, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Bebas Neue', sans-serif", fontSize: Math.round(size * 0.4), color: "#C9A84C" }}>
        {initials || "?"}
      </div>
    );
  }

  return (
    <img
      src={player.headshotUrl}
      alt={player.name}
      onError={() => setBroken(true)}
      style={{ ...baseStyle, objectFit: "cover" }}
    />
  );
}
