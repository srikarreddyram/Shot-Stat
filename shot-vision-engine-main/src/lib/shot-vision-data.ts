export interface Player {
  id: string;
  name: string;
  pos: string;
  heightIn: number;
  weightLbs: number;
  wingspanIn: number;
  offRtg: number;
  defRtgOvr: number;
  fg: number;
  tp: number;
  rim: number;
  defRtg: number;
  contest: string | number;
  ast?: number;
  tov?: number;
  ftPct?: number;
  rimPct?: number;
  midPct?: number;
}

// PLAYERS mock removed - now fetched from API

export interface Zone {
  key: string;
  label: string;
  shortLabel: string;
  baseEP: number;
  shotValue: number;
  color: string;
}

export const ZONES: Zone[] = [
  { key: "restrictedArea", label: "Restricted Area", shortLabel: "RA", baseEP: 1.36, shotValue: 2, color: "#16A34A" },
  { key: "paintNonRA", label: "In The Paint (Non-RA)", shortLabel: "PAINT", baseEP: 1.04, shotValue: 2, color: "#65A830" },
  { key: "leftCorner3", label: "Left Corner 3", shortLabel: "LC3", baseEP: 1.23, shotValue: 3, color: "#C9A84C" },
  { key: "rightCorner3", label: "Right Corner 3", shortLabel: "RC3", baseEP: 1.14, shotValue: 3, color: "#C9A84C" },
  { key: "aboveBreak3", label: "Above the Break 3", shortLabel: "AB3", baseEP: 1.08, shotValue: 3, color: "#B8860B" },
  { key: "midRange", label: "Mid-Range", shortLabel: "MID", baseEP: 0.88, shotValue: 2, color: "#DC2626" },
];

export interface EngineInput {
  attacker: Player;
  primaryDef: Player;
  secondaryDef: Player | null;
  doubleTeam: boolean;
  quarter: 1 | 2 | 3 | 4;
  minutes: number;
  seconds: number;
  scoreDiff: number;
  home: boolean;
}

export interface ZoneResult {
  zone: Zone;
  ep: number;
  makeProb: number;
}

// runEngine mock removed - now using API

export function heightToFeet(inches: number): string {
  const ft = Math.floor(inches / 12);
  const inc = inches % 12;
  return `${ft}'${inc}"`;
}