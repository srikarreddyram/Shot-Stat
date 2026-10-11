import { C, Card, F, GhostButton, NUM } from "./ui";

// A new season's first weeks on the Stat Engine. The engine stays on last
// season until teams have played a few games of the new one
// (src/inference/seasons.py), so the page says why it opened there and offers
// the new season; switched over, it says how little is behind the numbers and
// offers the way back. Between seasons, and once the new one is established
// (it is then the default), nothing shows.

/** /stats/players' season_status. */
export type SeasonStatus = {
  /** The season the engine opens on. */
  default: string;
  /** The newest season with any shots. */
  newest: string | null;
  newest_games: number;
  newest_games_per_team: number;
  /** Games a team that make the newest season the default. */
  established_games: number;
};

export function NewSeasonBanner({
  status,
  showing,
  onShow,
}: {
  status: SeasonStatus;
  showing: string;
  onShow: (season: string) => void;
}) {
  const { newest, default: settled, established_games: needed } = status;
  if (!newest || newest === settled) return null;
  const onNewest = showing === newest;
  const perTeam = status.newest_games_per_team;
  const played = (
    <>
      <span style={{ ...NUM, color: C.text }}>{status.newest_games}</span>{" "}
      {status.newest_games === 1 ? "game" : "games"} in,{" "}
      {perTeam < 1 ? "fewer than one a team" : `${Math.floor(perTeam)} a team`}
    </>
  );
  return (
    <Card
      accent={C.gold}
      style={{
        padding: "13px 18px",
        marginBottom: 18,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 14,
        flexWrap: "wrap",
      }}
    >
      <div style={{ flex: "1 1 420px", minWidth: 0 }}>
        <div
          style={{
            fontFamily: F.mono,
            fontSize: 10,
            letterSpacing: "0.3em",
            color: C.gold,
            marginBottom: 5,
          }}
        >
          {onNewest ? `${newest} SO FAR` : `${newest} HAS TIPPED OFF`}
        </div>
        <div style={{ fontSize: 12.5, color: C.dim, lineHeight: 1.55 }}>
          {onNewest ? (
            <>
              {played}. Ranks, tiers and ratings settle once teams have played {needed}; until then
              they move with every game.
            </>
          ) : (
            <>
              {played}: too few to rank anyone on, so the Stat Engine stays on {settled} until teams
              have played {needed}.
            </>
          )}
        </div>
      </div>
      <GhostButton onClick={() => onShow(onNewest ? settled : newest)}>
        {onNewest ? `← BACK TO ${settled}` : `SEE ${newest} SO FAR →`}
      </GhostButton>
    </Card>
  );
}
