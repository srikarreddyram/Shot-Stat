// Strips diacritics ("Dëmin" -> "Demin") so a client-side name filter
// matches the way it's typed. /players/search does this server-side via
// SQL's UNACCENT(); the team-vs-team roster picker filters a list it
// already has in memory, so it needs its own fold to behave the same way —
// without this, typing "demin" (no diaeresis) silently finds nothing for
// Egor Dëmin while the plain search box finds him fine.
//
// The pattern is built from character codes rather than written as a
// regex literal — U+0300-U+036F (combining diacritical marks) — because
// pasting the literal escape sequence into this file was, in this editing
// session, silently rewritten by the surrounding tooling into something
// that no longer matched. This form is inert text until run.
const DIACRITIC_MARKS = new RegExp(String.fromCharCode(0x5b, 0x5c, 0x75, 0x30, 0x33, 0x30, 0x30, 0x2d, 0x5c, 0x75, 0x30, 0x33, 0x36, 0x66, 0x5d), "g");

export function foldAccents(s: string): string {
  return s.normalize("NFD").replace(DIACRITIC_MARKS, "");
}
