/**
 * UNUSED as of the switch to video. SplashPage now renders video-backdrop.tsx
 * instead; this file is kept because the choreography and the glTF retarget
 * took real work to get right. To bring it back, import createCourtScene in
 * SplashPage and re-download the character model that public/models/player.glb
 * held:
 *
 *   curl -L -o public/models/player.glb \
 *     https://cdn.jsdelivr.net/gh/mrdoob/three.js@r157/examples/models/gltf/Xbot.glb
 *
 * The animated court behind the SHOT VISION splash page.
 *
 * Three procedurally-rigged figures run one continuous possession — live
 * dribble, closeout, kick-out pass, give-and-go cut, and a two-hand dunk.
 * Each scroll section owns one looping clip per figure plus a ball track.
 *
 * Clips are authored as *pose targets* (a flat bag of joint angles) rather
 * than baked animation data. That is the whole reason section changes look
 * smooth: switching sections cross-fades the weighted sum of the outgoing
 * and incoming poses instead of cutting between two animation states, so
 * limbs, root position and the ball all glide into the new clip.
 *
 * Sign conventions, since they are easy to get backwards:
 *   - Limb meshes hang along -Y from their joint, so +rotation.x swings the
 *     limb forward (toward -Z, which is where the hoop is).
 *   - Torso/head meshes extend along +Y, so a forward lean is -rotation.x.
 *   - Knee/elbow values in a Pose are *flex amounts*; the knee applies them
 *     negated because knees fold backwards.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { clone as cloneSkinned } from "three/examples/jsm/utils/SkeletonUtils.js";

/** Mixamo-rigged humanoid, vendored into public/. */
const PLAYER_MODEL_URL = "/models/player.glb";

export interface CourtScene {
  /** Cross-fade to the clip set for this scroll section. */
  setSection(index: number): void;
  dispose(): void;
}

/* ------------------------------------------------------------------ *
 * Court geometry constants (metres, real NBA dimensions)
 * ------------------------------------------------------------------ */

const BACKBOARD_Z = -7.0;
const RIM_Z = BACKBOARD_Z + 0.385;
const RIM_Y = 3.05;
const RIM_R = 0.2286;
const BALL_R = 0.122;
/** Pelvis height with legs straight; rootHeight() derives the real value. */
const LEG_HEIGHT = 1.035;
/** Chest height in a normal athletic stance — where a chest pass lives. */
const CHEST_Y = 1.42;

const A_HOME: V3 = [-1.25, 0, 1.7];
const A_CUT_END: V3 = [-0.4, 0, -3.2];
const T_SPOT: V3 = [2.6, 0, 0.7];
const D_HOME: V3 = [0.5, 0, 0.3];
const D_CLOSE: V3 = [-0.05, 0, 1.25];
const D_CHASE: V3 = [0.35, 0, -1.9];
const D_CONTEST: V3 = [0.75, 0, -4.9];

const SECTION_PERIOD = [2.4, 1.6, 2.8, 3.0, 4.2, 4.2];
const FADE_SECONDS = 0.9;

type V3 = [number, number, number];

/* ------------------------------------------------------------------ *
 * Pose model
 * ------------------------------------------------------------------ */

const REST = {
  rootX: 0,
  rootY: 0,
  rootZ: 0,
  rootRY: 0,
  pelvisX: 0,
  pelvisY: 0,
  chestX: -0.12,
  chestY: 0,
  chestZ: 0,
  headX: 0.08,
  headY: 0,
  sLX: 0,
  sLY: 0,
  sLZ: 0.36,
  eL: 0.5,
  sRX: 0,
  sRY: 0,
  sRZ: -0.36,
  eR: 0.5,
  hLX: 0.3,
  hLY: 0,
  hLZ: 0.09,
  kL: 0.6,
  hRX: 0.3,
  hRY: 0,
  hRZ: -0.09,
  kR: 0.6,
};

type Pose = typeof REST;
const POSE_KEYS = Object.keys(REST) as (keyof Pose)[];

const P = (o: Partial<Pose>): Pose => ({ ...REST, ...o });

const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);
const smooth = (t: number) => {
  const c = clamp01(t);
  return c * c * (3 - 2 * c);
};
/** Normalised progress through the sub-interval [a,b] of a clip. */
const seg = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const mix3 = (a: V3, b: V3, t: number): V3 => [
  mix(a[0], b[0], t),
  mix(a[1], b[1], t),
  mix(a[2], b[2], t),
];

/**
 * Pelvis height that puts the planted foot exactly on the floor for this
 * pose's leg angles, plus whatever airborne offset the pose asked for.
 * Using the *straighter* leg as the plant is what makes a run cycle bob
 * on its own instead of needing a hand-authored bounce.
 */
function rootHeight(p: Pose) {
  const reach = (h: number, k: number) => 0.155 + 0.44 * Math.cos(h) + 0.44 * Math.cos(h - k);
  return Math.max(reach(p.hLX, p.kL), reach(p.hRX, p.kR)) + p.rootY;
}

function blendPoses(a: Pose, b: Pose, t: number): Pose {
  if (t <= 0) return a;
  if (t >= 1) return b;
  const out = {} as Pose;
  for (const k of POSE_KEYS) out[k] = a[k] + (b[k] - a[k]) * t;
  return out;
}

/** Yaw that makes a figure (which faces -Z at rest) look along `dir`. */
function faceYaw(dx: number, dz: number) {
  return Math.atan2(-dx, -dz);
}

/** Symmetric running gait shared by every locomotion clip. */
function gait(phase: number, amp: number, pose: Partial<Pose>): Partial<Pose> {
  const s = Math.sin(phase * Math.PI * 2);
  return {
    hLX: 0.2 + s * amp,
    hRX: 0.2 - s * amp,
    kL: 0.55 + Math.max(0, -s) * amp * 1.6,
    kR: 0.55 + Math.max(0, s) * amp * 1.6,
    sLX: -s * amp * 0.8,
    sRX: s * amp * 0.8,
    eL: 0.9,
    eR: 0.9,
    ...pose,
  };
}

/* ------------------------------------------------------------------ *
 * Attacker clips (the gold figure — dribbles, cuts, finishes)
 * ------------------------------------------------------------------ */

/**
 * Live dribble. `cross` swaps the ball between hands every bounce instead of
 * pounding it with one hand, which is what section 1's guarded look needs.
 */
function aDribble(t: number, cross: boolean, guarded: boolean): Pose {
  const ph = (((cross ? t * 2 : t * 3) % 1) + 1) % 1;
  // 1 when the ball is up at the hand, 0 when it is on the floor.
  const up = Math.abs(Math.cos(Math.PI * ph));
  const side = cross ? Math.cos(Math.PI * 2 * t) : 1;
  const actR = cross ? clamp01(side) : 1;
  const actL = cross ? clamp01(-side) : 0;

  // An idle arm hangs; an active arm extends as it pushes the ball down.
  const armX = (act: number) => mix(0.12, mix(0.14, 0.5, up), act);
  const armE = (act: number) => mix(0.4, mix(0.25, 1.15, up), act);

  const dip = guarded ? 0.16 : 0.08;
  // Knees ride with the ball, which dips the whole body via rootHeight().
  const bend = 0.62 + dip * 2 + up * 0.16;
  return P({
    rootX: A_HOME[0],
    rootZ: A_HOME[2],
    rootRY: guarded ? -0.5 : -0.18 + Math.sin(t * Math.PI * 2) * 0.06,
    chestX: guarded ? -0.3 : -0.16,
    headX: 0.02,
    headY: guarded ? 0.35 : 0.1,
    hLX: 0.3 + dip,
    hRX: 0.3 + dip,
    kL: bend,
    kR: bend,
    hLZ: 0.16,
    hRZ: -0.16,
    sLX: armX(actL),
    eL: armE(actL),
    sLZ: 0.24 + actL * 0.18,
    sRX: armX(actR),
    eR: armE(actR),
    sRZ: -0.24 - actR * 0.18,
  });
}

/** Chest-height kick-out to the teammate on the wing. */
function aPass(t: number): Pose {
  const gather = smooth(seg(t, 0.0, 0.34));
  const push = smooth(seg(t, 0.34, 0.5));
  const relax = smooth(seg(t, 0.6, 1.0));
  const yaw = faceYaw(T_SPOT[0] - A_HOME[0], T_SPOT[2] - A_HOME[2]);
  const extend = push * (1 - relax);
  return P({
    rootX: A_HOME[0],
    rootZ: A_HOME[2],
    rootRY: mix(-0.5, yaw, gather * 0.9),
    chestX: mix(-0.24, -0.05, extend),
    chestY: mix(0.25, -0.15, extend),
    headY: 0.3,
    hLX: 0.4,
    hRX: 0.34,
    kL: 0.8,
    kR: 0.7,
    // Both arms load at the chest, then punch out together.
    sLX: mix(mix(0.2, 0.75, gather), 1.15, extend),
    sRX: mix(mix(0.2, 0.75, gather), 1.15, extend),
    eL: mix(mix(0.4, 1.75, gather), 0.15, extend),
    eR: mix(mix(0.4, 1.75, gather), 0.15, extend),
    sLZ: mix(0.3, 0.12, extend),
    sRZ: mix(-0.3, -0.12, extend),
  });
}

/** Where the attacker's root is during the section-3 give-and-go cut. */
function aCutPos(t: number): V3 {
  return mix3(A_HOME, A_CUT_END, smooth(seg(t, 0.05, 0.62)));
}

/** Cut to the rim, catch the return pass, take one dribble into the gather. */
function aCut(t: number): Pose {
  const pos = aCutPos(t);
  const run = smooth(seg(t, 0.05, 0.62));
  const catchT = smooth(seg(t, 0.52, 0.66));
  const carry = smooth(seg(t, 0.66, 1.0));
  const yaw = mix(faceYaw(0.6, -3.0), faceYaw(0.2, -1.0), catchT);
  return P({
    ...gait(t * 2, 0.62 * (1 - catchT * 0.6), {}),
    rootX: pos[0],
    rootZ: pos[2],
    rootRY: yaw,
    chestX: mix(-0.3, -0.18, catchT),
    headY: mix(0.5, 0.0, run),
    // Hands come up to receive, then drop to a one-hand carry.
    sLX: mix(mix(-0.4, 1.3, catchT), 0.45, carry),
    sRX: mix(mix(0.5, 1.3, catchT), 0.35, carry),
    eL: mix(mix(0.9, 0.5, catchT), 0.9, carry),
    eR: mix(mix(0.9, 0.5, catchT), 0.8, carry),
    sLZ: 0.26,
    sRZ: -0.26,
  });
}

/**
 * The dunk timeline, in clip-phase units. Shared by the body pose, the ball
 * track and the rim flex so they cannot drift out of sync.
 */
const DUNK = {
  driveA: 0.12,
  driveB: 0.4,
  gatherB: 0.48,
  riseB: 0.66,
  slamB: 0.74,
  fallA: 0.8,
  fallB: 0.92,
};

/** Root transform of the dunk, shared by the body clip and the ball track. */
function dunkPos(t: number): { x: number; y: number; z: number } {
  const drive = smooth(seg(t, DUNK.driveA, DUNK.driveB));
  const rise = smooth(seg(t, DUNK.gatherB, DUNK.riseB));
  const fall = smooth(seg(t, DUNK.fallA, DUNK.fallB));
  const back = smooth(seg(t, DUNK.fallB, 1.0));

  let x = mix(A_CUT_END[0], -0.1, drive);
  let z = mix(A_CUT_END[2], -5.1, drive);
  // The leap carries the last stride and a half, right under the rim.
  z = mix(z, -6.15, rise);
  x = mix(x, 0.0, rise);

  // Air only. The gather crouch comes out of the knee angles via
  // rootHeight(), and the hang holds near the peak until `fall` starts.
  let y = 1.5 * Math.sin(rise * Math.PI * 0.5) * (1 - fall);

  if (back > 0) {
    x = mix(x, A_CUT_END[0], back);
    z = mix(z, A_CUT_END[2], back);
    y = mix(y, 0, back);
  }
  return { x, y, z };
}

/** Drive, gather, two-hand slam, hang, land, reset. Loops as a highlight. */
function aDunk(t: number): Pose {
  const pos = dunkPos(t);
  const drive = seg(t, DUNK.driveA, DUNK.driveB);
  const gatherT = smooth(seg(t, DUNK.driveB, DUNK.gatherB));
  const rise = smooth(seg(t, DUNK.gatherB, DUNK.riseB));
  const slam = smooth(seg(t, DUNK.riseB, DUNK.slamB));
  const fall = smooth(seg(t, DUNK.fallA, DUNK.fallB));
  const back = smooth(seg(t, DUNK.fallB, 1.0));

  const driving = clamp01(drive) * (1 - gatherT);
  const bounce = Math.abs(Math.cos(Math.PI * drive * 2));

  // Reach over the rim, stay extended while the ball goes through, and only
  // drop the arms on the way back down.
  const armX = mix(mix(mix(0.5, 2.9, rise), 2.45, slam), 0.5, fall);
  const armE = mix(mix(mix(0.9, 0.08, rise), 0.28, slam), 0.9, fall);

  return P({
    rootX: pos.x,
    rootZ: pos.z,
    rootY: pos.y,
    rootRY: mix(faceYaw(0.4, -3.2), 0, rise) * (1 - back) + back * faceYaw(0, 1),
    chestX: mix(mix(-0.34, -0.55, gatherT), -0.3, rise),
    headX: mix(0.05, -0.4, rise),
    // Legs: sprint cycle -> deep gather -> tucked in the air -> absorb landing.
    hLX: mix(
      mix(0.2 + Math.sin(drive * 6 * Math.PI) * 0.6, 0.75, gatherT),
      mix(-0.35, 0.9, fall),
      rise,
    ),
    hRX: mix(
      mix(0.2 - Math.sin(drive * 6 * Math.PI) * 0.6, 0.75, gatherT),
      mix(0.55, 0.9, fall),
      rise,
    ),
    kL: mix(mix(0.6, 1.35, gatherT), mix(0.35, 1.1, fall), rise),
    kR: mix(mix(0.6, 1.35, gatherT), mix(1.1, 1.1, fall), rise),
    sLX: mix(mix(-0.3 * driving, 0.1, gatherT), armX, rise),
    sRX: mix(mix(0.3 + 0.35 * bounce * driving, 0.1, gatherT), armX, rise),
    eL: mix(mix(0.9, 1.5, gatherT), armE, rise),
    eR: mix(mix(mix(0.3, 1.1, bounce), 1.5, gatherT), armE, rise),
    sLZ: mix(0.34, 0.12, rise),
    sRZ: mix(-0.34, -0.12, rise),
  });
}

/* ------------------------------------------------------------------ *
 * Teammate clips (the pale figure on the wing — catch and give-and-go)
 * ------------------------------------------------------------------ */

function tIdle(t: number): Pose {
  const s = Math.sin(t * Math.PI * 2);
  return P({
    rootX: T_SPOT[0],
    rootZ: T_SPOT[2],
    rootRY: faceYaw(-3.0, -1.0) + s * 0.08,
    hLX: 0.34,
    hRX: 0.34,
    kL: 0.66,
    kR: 0.66,
    sLX: 0.2 + s * 0.1,
    sRX: 0.2 - s * 0.1,
    eL: 0.7,
    eR: 0.7,
    headY: -0.3,
  });
}

/** Hands up, ball arrives, gather to the chest. */
function tCatch(t: number): Pose {
  const reach = smooth(seg(t, 0.4, 0.7));
  const secure = smooth(seg(t, 0.72, 0.92));
  return P({
    rootX: T_SPOT[0],
    rootZ: T_SPOT[2],
    rootRY: faceYaw(-4.5, 1.6),
    chestX: mix(-0.12, -0.26, secure),
    hLX: 0.34,
    hRX: 0.34,
    kL: 0.66 + secure * 0.2,
    kR: 0.66 + secure * 0.2,
    sLX: mix(mix(0.2, 1.25, reach), 0.7, secure),
    sRX: mix(mix(0.2, 1.25, reach), 0.7, secure),
    eL: mix(mix(0.7, 0.25, reach), 1.7, secure),
    eR: mix(mix(0.7, 0.25, reach), 1.7, secure),
    sLZ: 0.22,
    sRZ: -0.22,
  });
}

/** Return pass to the cutter. */
function tReturn(t: number): Pose {
  const wind = smooth(seg(t, 0.0, 0.22));
  const push = smooth(seg(t, 0.22, 0.4));
  const relax = smooth(seg(t, 0.55, 1.0));
  const extend = push * (1 - relax);
  return P({
    rootX: T_SPOT[0],
    rootZ: T_SPOT[2],
    rootRY: faceYaw(A_CUT_END[0] - T_SPOT[0], A_CUT_END[2] - T_SPOT[2]),
    chestX: -0.14,
    chestY: mix(0.2, -0.12, extend),
    hLX: 0.36,
    hRX: 0.3,
    kL: 0.72,
    kR: 0.62,
    sLX: mix(mix(0.7, 0.9, wind), 1.2, extend),
    sRX: mix(mix(0.7, 0.9, wind), 1.2, extend),
    eL: mix(mix(1.7, 1.9, wind), 0.15, extend),
    eR: mix(mix(1.7, 1.9, wind), 0.15, extend),
    sLZ: mix(0.28, 0.1, extend),
    sRZ: mix(-0.28, -0.1, extend),
  });
}

/** Trailing the play with both arms up. */
function tWatch(t: number): Pose {
  const s = Math.sin(t * Math.PI * 2);
  return P({
    rootX: 5.2,
    rootZ: 0.2,
    rootRY: faceYaw(-5.2, -6.8),
    hLX: 0.26,
    hRX: 0.26,
    kL: 0.5,
    kR: 0.5,
    sLX: 2.4 + s * 0.12,
    sRX: 2.4 - s * 0.12,
    eL: 0.4,
    eR: 0.4,
    sLZ: 0.3,
    sRZ: -0.3,
    headX: -0.25,
  });
}

/* ------------------------------------------------------------------ *
 * Defender clips (the red figure)
 * ------------------------------------------------------------------ */

function dSlide(t: number): Pose {
  const s = Math.sin(t * Math.PI * 2);
  return P({
    rootX: D_HOME[0] + s * 0.7,
    rootZ: D_HOME[2] + Math.cos(t * Math.PI * 2) * 0.2,
    rootRY: faceYaw(A_HOME[0] - D_HOME[0], A_HOME[2] - D_HOME[2]),
    chestX: -0.22,
    hLX: 0.36,
    hRX: 0.36,
    hLZ: 0.3,
    hRZ: -0.3,
    kL: 0.86,
    kR: 0.86,
    // Wide, low hands — the defensive-slide silhouette.
    sLX: 0.3,
    sRX: 0.3,
    sLZ: 1.0,
    sRZ: -1.0,
    eL: 0.35,
    eR: 0.35,
  });
}

function dCloseout(t: number): Pose {
  const run = smooth(seg(t, 0.0, 0.55));
  const pos = mix3(D_HOME, D_CLOSE, run);
  const hand = smooth(seg(t, 0.5, 0.8));
  const chop = Math.abs(Math.sin(t * Math.PI * 4)) * (1 - hand);
  return P({
    rootX: pos[0],
    rootZ: pos[2],
    rootRY: faceYaw(A_HOME[0] - pos[0], A_HOME[2] - pos[2]),
    rootY: -chop * 0.04,
    chestX: mix(-0.3, -0.16, hand),
    hLX: mix(0.2 + Math.sin(t * 8 * Math.PI) * 0.55, 0.4, hand),
    hRX: mix(0.2 - Math.sin(t * 8 * Math.PI) * 0.55, 0.4, hand),
    kL: mix(0.6, 0.9, hand),
    kR: mix(0.6, 0.9, hand),
    hLZ: 0.26,
    hRZ: -0.26,
    // High contest hand goes up as the feet chop down.
    sLX: mix(-0.5, 0.4, hand),
    sRX: mix(0.5, 2.75, hand),
    eL: mix(0.9, 0.4, hand),
    eR: mix(0.9, 0.12, hand),
    sLZ: mix(0.24, 0.9, hand),
    sRZ: -0.24,
  });
}

function dChase(t: number, from: V3, to: V3, jump: boolean): Pose {
  const run = smooth(seg(t, 0.05, jump ? 0.5 : 0.75));
  const pos = mix3(from, to, run);
  const leap = jump ? Math.sin(smooth(seg(t, 0.55, 0.85)) * Math.PI) : 0;
  const moving = from[0] !== to[0] || from[2] !== to[2];
  const cyc = moving ? t * 2.2 : t;
  const s = Math.sin(cyc * Math.PI * 2) * (moving ? 1 : 0.25);
  const amp = 0.7 * (1 - leap);
  return P({
    rootX: pos[0],
    rootZ: pos[2],
    rootRY: moving ? faceYaw(to[0] - from[0], to[2] - from[2]) : faceYaw(0.2, -1),
    rootY: leap * 1.05,
    chestX: -0.3,
    hLX: mix(0.2 + s * amp, -0.2, leap),
    hRX: mix(0.2 - s * amp, 0.3, leap),
    kL: mix(0.6 + Math.max(0, -s) * amp * 1.6, 0.9, leap),
    kR: mix(0.6 + Math.max(0, s) * amp * 1.6, 0.5, leap),
    // Arms pump on the run, then both go straight up on the (late) contest.
    sLX: mix(-s * amp * 0.8, 2.8, leap),
    sRX: mix(s * amp * 0.8, 2.8, leap),
    eL: mix(0.9, 0.1, leap),
    eR: mix(0.9, 0.1, leap),
  });
}

/* ------------------------------------------------------------------ *
 * Clip tables — index is the scroll section
 * ------------------------------------------------------------------ */

const ATTACKER_CLIPS: ((t: number) => Pose)[] = [
  (t) => aDribble(t, false, false),
  (t) => aDribble(t, true, true),
  aPass,
  aCut,
  aDunk,
  aDunk,
];

const TEAMMATE_CLIPS: ((t: number) => Pose)[] = [tIdle, tIdle, tCatch, tReturn, tWatch, tWatch];

const DEFENDER_CLIPS: ((t: number) => Pose)[] = [
  dSlide,
  dCloseout,
  (t) => dChase(t, D_CLOSE, D_CLOSE, false),
  (t) => dChase(t, D_CLOSE, D_CHASE, false),
  (t) => dChase(t, D_CHASE, D_CONTEST, true),
  (t) => dChase(t, D_CHASE, D_CONTEST, true),
];

/* ------------------------------------------------------------------ *
 * Ball tracks — world-space position per section
 * ------------------------------------------------------------------ */

/** A bounce between `top` and the floor, `count` times across the clip. */
function bounceY(t: number, count: number, top: number) {
  const ph = (((t * count) % 1) + 1) % 1;
  return BALL_R + (top - BALL_R) * Math.abs(Math.cos(Math.PI * ph));
}

/** Parabolic pass flight with a modest apex. */
function passArc(from: V3, to: V3, t: number, apex: number): V3 {
  const p = mix3(from, to, t);
  return [p[0], p[1] + Math.sin(t * Math.PI) * apex, p[2]];
}

const A_CHEST: V3 = [A_HOME[0] + 0.15, CHEST_Y - 0.12, A_HOME[2] - 0.1];
const T_CHEST: V3 = [T_SPOT[0] - 0.2, CHEST_Y - 0.15, T_SPOT[2] + 0.05];

function ballTrack(section: number, t: number): V3 {
  switch (section) {
    case 0:
      return [A_HOME[0] + 0.46, bounceY(t, 3, 1.02), A_HOME[2] + 0.18];
    case 1: {
      // Crossover: the ball swings side to side and bounces twice per swing.
      const side = Math.cos(Math.PI * 2 * t);
      return [A_HOME[0] + side * 0.5, bounceY(t, 2, 0.92), A_HOME[2] + 0.3];
    }
    case 2: {
      const gather = smooth(seg(t, 0.1, 0.34));
      const flight = seg(t, 0.42, 0.74);
      const start: V3 = [A_HOME[0] + 0.46, 1.02, A_HOME[2] + 0.18];
      const held = mix3(start, A_CHEST, gather);
      if (flight <= 0) return held;
      return passArc(A_CHEST, T_CHEST, smooth(flight), 0.28);
    }
    case 3: {
      const flight = seg(t, 0.24, 0.56);
      if (flight <= 0) return T_CHEST;
      const cut = aCutPos(0.56);
      const target: V3 = [cut[0] + 0.3, CHEST_Y - 0.1, cut[2] + 0.1];
      if (flight < 1) return passArc(T_CHEST, target, smooth(flight), 0.3);
      // Carried into one dribble as he closes on the rim.
      const carry = seg(t, 0.56, 1.0);
      const pos = aCutPos(mix(0.56, 1.0, carry));
      return [
        pos[0] + 0.42,
        mix(CHEST_Y - 0.1, bounceY(carry, 1, 1.0), smooth(carry)),
        pos[2] + 0.2,
      ];
    }
    default: {
      // Dunk: dribbled in, gathered to the hip, carried overhead, driven
      // through the rim, then bounced away under the basket.
      const pose = aDunk(t);
      const rootY = rootHeight(pose);
      const drive = seg(t, DUNK.driveA, DUNK.driveB);
      const gather = smooth(seg(t, DUNK.driveB, DUNK.gatherB));
      const rise = smooth(seg(t, DUNK.gatherB, DUNK.riseB));
      const slam = smooth(seg(t, DUNK.riseB, DUNK.slamB));
      const drop = seg(t, DUNK.slamB, DUNK.fallB);

      if (drop > 0) {
        // Through the net, onto the floor, and a bounce under the rim.
        const y =
          mix(RIM_Y - 0.15, BALL_R, smooth(drop)) +
          Math.abs(Math.sin(drop * Math.PI * 2)) * 0.3 * drop;
        const loose: V3 = [0, y, RIM_Z + 0.15];
        // Scooped back up during the backpedal so the loop has no seam.
        const regather = seg(t, DUNK.fallB, 1.0);
        if (regather <= 0) return loose;
        return mix3(loose, [pose.rootX - 0.44, 0.95, pose.rootZ + 0.18], smooth(regather));
      }
      const hip: V3 = [pose.rootX - 0.36, rootY + 0.15, pose.rootZ + 0.12];
      const overhead: V3 = [pose.rootX - 0.16, rootY + 1.3, pose.rootZ - 0.34];
      const atRim: V3 = [0, RIM_Y + 0.16, RIM_Z + 0.02];

      const carried = mix3(
        [pose.rootX - 0.44, bounceY(drive, 2, 1.0), pose.rootZ + 0.18],
        hip,
        gather,
      );
      const raised = mix3(carried, overhead, rise);
      return mix3(raised, atRim, slam);
    }
  }
}

/** How hard the rim is flexing (0..1) — only the dunk sections drive it. */
function rimFlexAt(section: number, t: number) {
  if (section < 4) return 0;
  const hit = seg(t, DUNK.riseB + 0.04, DUNK.slamB);
  const release = seg(t, DUNK.slamB, DUNK.fallB + 0.05);
  return Math.sin(hit * Math.PI * 0.5) * (1 - smooth(release));
}

/* ------------------------------------------------------------------ *
 * Camera framing per section
 * ------------------------------------------------------------------ */

const CAMERAS: { pos: V3; look: V3 }[] = [
  // Copy centred: aim high so the players settle into the lower third.
  { pos: [1.4, 3.3, 8.6], look: [-1.0, 3.1, 1.5] },
  // Copy left: closeout framed right of centre, full figures in frame.
  { pos: [-4.2, 2.6, 7.0], look: [1.6, 2.1, 1.6] },
  // Copy right: aim past the receiver so the whole pass lane sits left.
  { pos: [4.6, 2.9, 8.2], look: [2.2, 2.2, 0.0] },
  // Copy left: the cut runs up the right half.
  { pos: [8.0, 3.2, 1.0], look: [0.0, 2.2, 2.0] },
  // Copy right: side-on from the weak side, so the whole drive stays in
  // frame and the rim sits left of the headline.
  { pos: [-6.4, 3.0, -4.6], look: [-1.5, 2.4, -3.4] },
  // CTA centred: same angle, aimed high so the finish lands below the copy.
  { pos: [-3.0, 2.8, 1.0], look: [-1.0, 2.5, -0.4] },
];
/* ------------------------------------------------------------------ *
 * Scene construction
 * ------------------------------------------------------------------ */

/**
 * The invisible skeleton the clips drive. It carries no meshes — only joint
 * nodes whose world positions give the direction of each limb segment, which
 * is what the retarget aims the real character's bones along.
 */
type SourceRig = {
  root: THREE.Object3D;
  pelvis: THREE.Object3D;
  chest: THREE.Object3D;
  head: THREE.Object3D;
  headTop: THREE.Object3D;
  shL: THREE.Object3D;
  elL: THREE.Object3D;
  handL: THREE.Object3D;
  shR: THREE.Object3D;
  elR: THREE.Object3D;
  handR: THREE.Object3D;
  hipL: THREE.Object3D;
  kneeL: THREE.Object3D;
  footL: THREE.Object3D;
  hipR: THREE.Object3D;
  kneeR: THREE.Object3D;
  footR: THREE.Object3D;
};

function buildSourceRig(): SourceRig {
  const node = (x = 0, y = 0, z = 0) => {
    const o = new THREE.Object3D();
    o.position.set(x, y, z);
    return o;
  };
  // Segment lengths match the glTF rig's proportions so the derived pelvis
  // height in rootHeight() lines up with the character's actual legs.
  const root = node();
  const pelvis = node();
  root.add(pelvis);
  const chest = node(0, 0.5, 0);
  pelvis.add(chest);
  const head = node(0, 0.13, 0);
  chest.add(head);
  const headTop = node(0, 0.26, 0);
  head.add(headTop);

  const arm = (side: 1 | -1) => {
    const sh = node(0.225 * side, 0.06, 0);
    const el = node(0, -0.3, 0);
    const hand = node(0, -0.29, 0);
    el.add(hand);
    sh.add(el);
    chest.add(sh);
    return { sh, el, hand };
  };
  const leg = (side: 1 | -1) => {
    const hip = node(0.115 * side, -0.06, 0);
    const knee = node(0, -0.44, 0);
    const foot = node(0, -0.44, 0);
    knee.add(foot);
    hip.add(knee);
    pelvis.add(hip);
    return { hip, knee, foot };
  };
  const aL = arm(1),
    aR = arm(-1),
    lL = leg(1),
    lR = leg(-1);
  return {
    root,
    pelvis,
    chest,
    head,
    headTop,
    shL: aL.sh,
    elL: aL.el,
    handL: aL.hand,
    shR: aR.sh,
    elR: aR.el,
    handR: aR.hand,
    hipL: lL.hip,
    kneeL: lL.knee,
    footL: lL.foot,
    hipR: lR.hip,
    kneeR: lR.knee,
    footR: lR.foot,
  };
}

function applyPose(fig: SourceRig, p: Pose) {
  fig.root.position.set(p.rootX, rootHeight(p), p.rootZ);
  fig.root.rotation.y = p.rootRY;
  fig.pelvis.rotation.set(p.pelvisX, p.pelvisY, 0);
  fig.chest.rotation.set(p.chestX, p.chestY, p.chestZ);
  fig.head.rotation.set(p.headX, p.headY, 0);
  fig.shL.rotation.set(p.sLX, p.sLY, p.sLZ);
  fig.shR.rotation.set(p.sRX, p.sRY, p.sRZ);
  fig.elL.rotation.x = p.eL;
  fig.elR.rotation.x = p.eR;
  fig.hipL.rotation.set(p.hLX, p.hLY, p.hLZ);
  fig.hipR.rotation.set(p.hRX, p.hRY, p.hRZ);
  // Knees fold backwards, hence the negation.
  fig.kneeL.rotation.x = -p.kL;
  fig.kneeR.rotation.x = -p.kR;
  fig.root.updateMatrixWorld(true);
}

/* ------------------------------------------------------------------ *
 * Retargeting the source rig onto the glTF character
 * ------------------------------------------------------------------ */

/**
 * Which source segment drives which bone. The direction of the segment
 * (joint -> its child joint) is what the bone gets aimed along, so the
 * mapping never has to know Mixamo's bind orientations: arm bones point
 * down local +X and leg bones down local -Y, and aiming absorbs both.
 */
const boneKey = (name: string) => name.toLowerCase().replace(/[^a-z0-9]/g, "");

const BONE_AIM: { bone: string; from: keyof SourceRig; to: keyof SourceRig }[] = [
  { bone: "mixamorig:Spine1", from: "chest", to: "head" },
  { bone: "mixamorig:Head", from: "head", to: "headTop" },
  { bone: "mixamorig:LeftArm", from: "shL", to: "elL" },
  { bone: "mixamorig:LeftForeArm", from: "elL", to: "handL" },
  { bone: "mixamorig:RightArm", from: "shR", to: "elR" },
  { bone: "mixamorig:RightForeArm", from: "elR", to: "handR" },
  { bone: "mixamorig:LeftUpLeg", from: "hipL", to: "kneeL" },
  { bone: "mixamorig:LeftLeg", from: "kneeL", to: "footL" },
  { bone: "mixamorig:RightUpLeg", from: "hipR", to: "kneeR" },
  { bone: "mixamorig:RightLeg", from: "kneeR", to: "footR" },
];

type Character = {
  group: THREE.Group;
  hips: THREE.Bone;
  /** Every bone, ordered root-first, so world rotations accumulate in one pass. */
  ordered: THREE.Bone[];
  rest: Map<THREE.Bone, THREE.Quaternion>;
  /** Normalised direction to each aimed bone's child, in that bone's own space. */
  childDir: Map<THREE.Bone, THREE.Vector3>;
  aimOf: Map<THREE.Bone, { from: THREE.Object3D; to: THREE.Object3D }>;
  /** Pelvis height above the group origin at bind, after scaling. */
  pelvisRestY: number;
  worldQ: Map<THREE.Bone, THREE.Quaternion>;
};

function makeCharacter(
  source: THREE.Group,
  rig: SourceRig,
  color: number,
  emissive: number,
): Character {
  const group = cloneSkinned(source) as unknown as THREE.Group;

  // Team colours. Materials are shared across clones, so they must be copied
  // before tinting or all three players come out the same colour.
  group.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    const tinted = mats.map((m) => {
      const c = (m as THREE.MeshStandardMaterial).clone();
      c.color = new THREE.Color(color);
      c.emissive = new THREE.Color(emissive);
      c.emissiveIntensity = 0.14;
      c.roughness = 0.55;
      c.metalness = 0.05;
      return c;
    });
    mesh.material = tinted.length === 1 ? tinted[0] : tinted;
    // Limbs regularly swing outside the mesh's original bounds once the
    // skeleton is driven, and culling on stale bounds pops whole players out.
    mesh.frustumCulled = false;
  });

  const bones = new Map<string, THREE.Bone>();
  const ordered: THREE.Bone[] = [];
  const rest = new Map<THREE.Bone, THREE.Quaternion>();
  // traverse() is depth-first pre-order, so parents always land before children.
  group.traverse((o) => {
    const b = o as THREE.Bone;
    if (!b.isBone) return;
    bones.set(boneKey(b.name), b);
    ordered.push(b);
    rest.set(b, b.quaternion.clone());
  });

  const hips = bones.get(boneKey("mixamorig:Hips"));
  if (!hips) throw new Error(`no Hips bone; saw: ${[...bones.keys()].slice(0, 8).join(", ")}`);

  // Scale from the model's own bind pose rather than assuming its units, so
  // the pelvis lands exactly where rootHeight() puts it and feet meet the floor.
  group.updateMatrixWorld(true);
  const probe = new THREE.Vector3();
  hips.getWorldPosition(probe);
  group.scale.multiplyScalar(LEG_HEIGHT / probe.y);
  group.updateMatrixWorld(true);
  hips.getWorldPosition(probe);
  const pelvisRestY = probe.y;

  const childDir = new Map<THREE.Bone, THREE.Vector3>();
  const aimOf = new Map<THREE.Bone, { from: THREE.Object3D; to: THREE.Object3D }>();
  for (const m of BONE_AIM) {
    const bone = bones.get(boneKey(m.bone));
    if (!bone) continue;
    const child = bone.children.find((c) => (c as THREE.Bone).isBone) as THREE.Bone | undefined;
    if (!child || child.position.lengthSq() === 0) continue;
    childDir.set(bone, child.position.clone().normalize());
    aimOf.set(bone, { from: rig[m.from] as THREE.Object3D, to: rig[m.to] as THREE.Object3D });
  }

  const worldQ = new Map<THREE.Bone, THREE.Quaternion>();
  for (const b of ordered) worldQ.set(b, new THREE.Quaternion());
  return { group, hips, ordered, rest, childDir, aimOf, pelvisRestY, worldQ };
}

const _q = new THREE.Quaternion();
const _qi = new THREE.Quaternion();
const _v = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _ident = new THREE.Quaternion();

/**
 * Drive the character's skeleton from the (already posed) source rig.
 *
 * One root-first pass: each bone is solved using the world rotation its
 * parent got earlier in the same pass, then publishes its own. Doing it this
 * way avoids updateMatrixWorld() per bone, which would be O(n^2) over a
 * 67-bone rig every frame for every player — and avoids the frame-stale
 * parent rotations that a separate accumulation pass would hand back.
 */
function retarget(ch: Character, rig: SourceRig) {
  rig.root.getWorldPosition(_v);
  ch.group.position.set(_v.x, _v.y - ch.pelvisRestY, _v.z);

  rig.pelvis.getWorldQuaternion(_q);

  for (const bone of ch.ordered) {
    const parent = bone.parent as THREE.Bone | null;
    const pw = parent && parent.isBone ? ch.worldQ.get(parent)! : _ident;
    const restQ = ch.rest.get(bone)!;

    if (bone === ch.hips) {
      // Hips carry the body's world orientation as a delta on the bind pose.
      bone.quaternion.copy(_q).multiply(restQ);
    } else {
      const aim = ch.aimOf.get(bone);
      if (aim) {
        aim.from.getWorldPosition(_a);
        aim.to.getWorldPosition(_b);
        _dir.subVectors(_b, _a);
        if (_dir.lengthSq() > 1e-8) {
          // Target direction expressed in the parent's frame, then matched
          // against where this bone's child points at bind.
          _qi.copy(pw).invert();
          _dir.normalize().applyQuaternion(_qi);
          _v.copy(ch.childDir.get(bone)!).applyQuaternion(restQ);
          bone.quaternion.setFromUnitVectors(_v, _dir).multiply(restQ);
        }
      }
    }
    ch.worldQ.get(bone)!.copy(pw).multiply(bone.quaternion);
  }
}

/**
 * The arena around the court. Without this the top half of every frame is
 * flat clear-colour black, which reads as "unfinished 3D scene" rather than
 * as a dark arena. A gradient shell plus a catwalk of soft overhead glows
 * gives the void some depth for effectively no cost.
 */
function buildArena(scene: THREE.Scene) {
  const shell = new THREE.Mesh(
    new THREE.SphereGeometry(58, 24, 16),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
      uniforms: {
        horizon: { value: new THREE.Color(0x0a0a0f) },
        upper: { value: new THREE.Color(0x1c2333) },
        glow: { value: new THREE.Color(0x2a2214) },
      },
      vertexShader: `
        varying float vH;
        void main() {
          vH = normalize(position).y;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: `
        uniform vec3 horizon; uniform vec3 upper; uniform vec3 glow;
        varying float vH;
        void main() {
          // Warm haze just above the floor, cool dark rafters overhead.
          float up = smoothstep(0.0, 0.55, vH);
          vec3 c = mix(horizon, upper, up);
          c += glow * (1.0 - smoothstep(0.0, 0.22, abs(vH - 0.04)));
          gl_FragColor = vec4(c, 1.0);
        }`,
    }),
  );
  scene.add(shell);

  // Overhead light rig — additive sprites so they bloom softly.
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, "rgba(255,244,214,1)");
  g.addColorStop(0.35, "rgba(255,232,180,0.35)");
  g.addColorStop(1, "rgba(255,225,160,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  const spriteMat = new THREE.SpriteMaterial({
    map: tex,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    transparent: true,
    opacity: 0.32,
    fog: false,
  });
  for (let i = 0; i < 14; i++) {
    const sp = new THREE.Sprite(spriteMat);
    const col = i % 7;
    const row = Math.floor(i / 7);
    sp.position.set(-9 + col * 3, 8.5 + row * 1.2, -10 + row * 7);
    sp.scale.setScalar(1.5);
    scene.add(sp);
  }
}

function buildCourt(scene: THREE.Scene) {
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(46, 46),
    new THREE.MeshStandardMaterial({ color: 0x7a4a20, roughness: 0.82, metalness: 0.05 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.02;
  scene.add(floor);

  const lineMat = new THREE.LineBasicMaterial({
    color: 0xe8c88a,
    transparent: true,
    opacity: 0.85,
  });
  const addLine = (pts: number[][]) => {
    const geo = new THREE.BufferGeometry().setFromPoints(
      pts.map(([x, z]) => new THREE.Vector3(x, 0.005, z)),
    );
    scene.add(new THREE.Line(geo, lineMat));
  };

  const baseline = -7.31;
  // Paint + free-throw line.
  addLine([
    [-2.44, baseline],
    [-2.44, baseline + 5.79],
    [2.44, baseline + 5.79],
    [2.44, baseline],
  ]);
  // Free-throw circle.
  const ft: number[][] = [];
  for (let i = 0; i <= 60; i++) {
    const a = (i / 60) * Math.PI * 2;
    ft.push([Math.cos(a) * 1.83, baseline + 5.79 + Math.sin(a) * 1.83]);
  }
  addLine(ft);
  // Three-point arc plus the two corner straightaways.
  const arc: number[][] = [];
  for (let i = 0; i <= 80; i++) {
    const a = mix(-Math.PI * 0.5, Math.PI * 0.5, i / 80);
    const x = Math.sin(a) * 7.24;
    const z = RIM_Z + Math.cos(a) * 7.24;
    if (Math.abs(x) <= 6.7) arc.push([x, z]);
  }
  addLine(arc);
  addLine([
    [-6.7, baseline],
    [-6.7, arc[0][1]],
  ]);
  addLine([
    [6.7, baseline],
    [6.7, arc[arc.length - 1][1]],
  ]);
  addLine([
    [-7.5, baseline],
    [7.5, baseline],
  ]);
  // Restricted-area semicircle.
  const ra: number[][] = [];
  for (let i = 0; i <= 40; i++) {
    const a = mix(-Math.PI * 0.5, Math.PI * 0.5, i / 40);
    ra.push([Math.sin(a) * 1.22, RIM_Z + Math.cos(a) * 1.22]);
  }
  addLine(ra);
}

function buildHoop(scene: THREE.Scene) {
  const glass = new THREE.MeshStandardMaterial({
    color: 0x9fb4c8,
    transparent: true,
    opacity: 0.16,
    roughness: 0.1,
    metalness: 0.4,
  });
  const board = new THREE.Mesh(new THREE.BoxGeometry(1.83, 1.07, 0.04), glass);
  board.position.set(0, 3.6, BACKBOARD_Z);
  scene.add(board);

  const trim = new THREE.LineBasicMaterial({ color: 0xf0f0f0, transparent: true, opacity: 0.55 });
  const rect = (w: number, h: number, y: number) => {
    const pts = [
      [-w / 2, y - h / 2],
      [w / 2, y - h / 2],
      [w / 2, y + h / 2],
      [-w / 2, y + h / 2],
      [-w / 2, y - h / 2],
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(
      pts.map(([x, yy]) => new THREE.Vector3(x, yy, BACKBOARD_Z + 0.025)),
    );
    scene.add(new THREE.Line(geo, trim));
  };
  rect(1.83, 1.07, 3.6);
  rect(0.59, 0.45, 3.27);

  // Rim + net live in a group pivoted at the backboard so the whole
  // assembly can flex downward when the ball is driven through it.
  const rimGroup = new THREE.Group();
  rimGroup.position.set(0, RIM_Y, BACKBOARD_Z);
  scene.add(rimGroup);

  const rimMat = new THREE.MeshStandardMaterial({
    color: 0xe0521a,
    emissive: 0xc8400f,
    emissiveIntensity: 0.35,
    roughness: 0.4,
    metalness: 0.7,
  });
  const rim = new THREE.Mesh(new THREE.TorusGeometry(RIM_R, 0.017, 10, 44), rimMat);
  rim.rotation.x = -Math.PI / 2;
  rim.position.z = 0.385;
  rimGroup.add(rim);
  const mount = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.05, 0.18), rimMat);
  mount.position.z = 0.08;
  rimGroup.add(mount);

  const netMat = new THREE.LineBasicMaterial({ color: 0xe8e8ee, transparent: true, opacity: 0.42 });
  const net = new THREE.Group();
  net.position.z = 0.385;
  const STRANDS = 14;
  for (let i = 0; i < STRANDS; i++) {
    const a = (i / STRANDS) * Math.PI * 2;
    const pts = [];
    for (let k = 0; k <= 4; k++) {
      const f = k / 4;
      const r = mix(RIM_R, RIM_R * 0.56, f);
      // Slight twist down the net so the strands read as woven, not straight.
      const aa = a + f * 0.35;
      pts.push(new THREE.Vector3(Math.cos(aa) * r, -0.45 * f, Math.sin(aa) * r));
    }
    net.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), netMat));
  }
  rimGroup.add(net);

  // Stanchion.
  const poleMat = new THREE.MeshStandardMaterial({
    color: 0x14141c,
    roughness: 0.6,
    metalness: 0.5,
  });
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.14, 3.6, 12), poleMat);
  pole.position.set(0, 1.8, BACKBOARD_Z - 0.95);
  scene.add(pole);
  const arm = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.14, 0.95), poleMat);
  arm.position.set(0, 3.55, BACKBOARD_Z - 0.5);
  scene.add(arm);
  const base = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.16, 1.2), poleMat);
  base.position.set(0, 0.08, BACKBOARD_Z - 1.1);
  scene.add(base);

  return { rimGroup, net };
}

/* ------------------------------------------------------------------ *
 * Entry point
 * ------------------------------------------------------------------ */

export async function createCourtScene(canvas: HTMLCanvasElement): Promise<CourtScene> {
  let disposed = false;

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x0a0a0f);

  const scene = new THREE.Scene();
  // Deep enough to swallow the far edge of the floor plane, shallow enough
  // that the hoop at ~7m only picks up a little haze.
  scene.fog = new THREE.FogExp2(0x0a0a0f, 0.055);

  const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 120);
  camera.position.set(CAMERAS[0].pos[0], CAMERAS[0].pos[1], CAMERAS[0].pos[2]);

  buildArena(scene);
  buildCourt(scene);
  const { rimGroup, net } = buildHoop(scene);

  // Each player is an invisible source rig (which the clips drive) plus a
  // skinned character retargeted onto it every frame.
  const rigs = {
    attacker: buildSourceRig(),
    teammate: buildSourceRig(),
    defender: buildSourceRig(),
  };
  scene.add(rigs.attacker.root, rigs.teammate.root, rigs.defender.root);

  let characters: { attacker: Character; teammate: Character; defender: Character } | null = null;
  const TEAM = {
    attacker: [0xc9a84c, 0xc9a84c] as const,
    teammate: [0xdfe3ec, 0x8fa6c8] as const,
    defender: [0xdc2626, 0xdc2626] as const,
  };

  // The model is ~2.9MB, so the scene starts rendering the court and only
  // pops the players in once it lands rather than blocking the whole splash.
  new GLTFLoader().load(
    PLAYER_MODEL_URL,
    (gltf) => {
      if (disposed) return;
      const src = gltf.scene;
      characters = {
        attacker: makeCharacter(src, rigs.attacker, ...TEAM.attacker),
        teammate: makeCharacter(src, rigs.teammate, ...TEAM.teammate),
        defender: makeCharacter(src, rigs.defender, ...TEAM.defender),
      };
      scene.add(characters.attacker.group, characters.teammate.group, characters.defender.group);
    },
    undefined,
    (err) => {
      // Never silent: a failed character load must still leave a trace, or
      // the scene just quietly renders an empty court.
      console.error("[court-scene] player model failed", err);
    },
  );

  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(BALL_R, 28, 28),
    new THREE.MeshStandardMaterial({
      color: 0xc85a00,
      emissive: 0x662200,
      emissiveIntensity: 0.25,
      roughness: 0.85,
    }),
  );
  scene.add(ball);

  // Lighting — warm arena key, cool fill, spot over the rim.
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x4a2a10, 0.5));
  const key = new THREE.DirectionalLight(0xffe9b8, 1.7);
  key.position.set(5, 8, 4);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x6f90ff, 0.7);
  fill.position.set(-6, 4, 3);
  scene.add(fill);
  const rimLight = new THREE.SpotLight(0xffffff, 1.6, 22, Math.PI / 5, 0.5);
  rimLight.position.set(0, 9, BACKBOARD_Z + 2);
  rimLight.target.position.set(0, RIM_Y, RIM_Z);
  scene.add(rimLight, rimLight.target);
  [
    [9, 8, 6],
    [-9, 8, 6],
    [9, 8, -6],
    [-9, 8, -6],
  ].forEach(([x, y, z]) => {
    const p = new THREE.PointLight(0xfff5cc, 0.3, 45);
    p.position.set(x, y, z);
    scene.add(p);
  });

  const resize = () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  resize();
  window.addEventListener("resize", resize);

  const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;

  let current = 0;
  let previous = 0;
  let fade = 1;
  let raf = 0;
  let last = performance.now();
  let elapsed = 0;

  const camPos = new THREE.Vector3(...CAMERAS[0].pos);
  const camLook = new THREE.Vector3(...CAMERAS[0].look);
  const tmpPos = new THREE.Vector3();
  const tmpLook = new THREE.Vector3();
  const prevBall = new THREE.Vector3();

  const phaseOf = (section: number) =>
    (elapsed % SECTION_PERIOD[section]) / SECTION_PERIOD[section];

  const frame = (now: number) => {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (!reduced) elapsed += dt;
    if (fade < 1) fade = Math.min(1, fade + dt / FADE_SECONDS);
    const w = smooth(fade);

    const tPrev = phaseOf(previous);
    const tCur = phaseOf(current);

    applyPose(
      rigs.attacker,
      blendPoses(ATTACKER_CLIPS[previous](tPrev), ATTACKER_CLIPS[current](tCur), w),
    );
    applyPose(
      rigs.teammate,
      blendPoses(TEAMMATE_CLIPS[previous](tPrev), TEAMMATE_CLIPS[current](tCur), w),
    );
    applyPose(
      rigs.defender,
      blendPoses(DEFENDER_CLIPS[previous](tPrev), DEFENDER_CLIPS[current](tCur), w),
    );
    if (characters) {
      retarget(characters.attacker, rigs.attacker);
      retarget(characters.teammate, rigs.teammate);
      retarget(characters.defender, rigs.defender);
    }

    const bPrev = ballTrack(previous, tPrev);
    const bCur = ballTrack(current, tCur);
    const b = mix3(bPrev, bCur, w);
    prevBall.set(ball.position.x, ball.position.y, ball.position.z);
    ball.position.set(b[0], b[1], b[2]);
    // Spin the ball along its travel direction so it never looks pasted on.
    ball.rotation.x -= (ball.position.z - prevBall.z) / BALL_R;
    ball.rotation.z += (ball.position.x - prevBall.x) / BALL_R;

    const flex = mix(rimFlexAt(previous, tPrev), rimFlexAt(current, tCur), w);
    rimGroup.rotation.x = flex * 0.22;
    net.scale.y = 1 + flex * 0.5;

    const cam = CAMERAS[current];
    const camPrev = CAMERAS[previous];
    tmpPos.set(...(mix3(camPrev.pos, cam.pos, w) as [number, number, number]));
    tmpLook.set(...(mix3(camPrev.look, cam.look, w) as [number, number, number]));
    // A slow sway keeps the framing alive between sections.
    tmpPos.x += Math.sin(elapsed * 0.22) * 0.28;
    tmpPos.y += Math.cos(elapsed * 0.17) * 0.12;
    camPos.lerp(tmpPos, 1 - Math.pow(0.001, dt));
    camLook.lerp(tmpLook, 1 - Math.pow(0.001, dt));
    camera.position.copy(camPos);
    camera.lookAt(camLook);

    renderer.render(scene, camera);
    if (!disposed) raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);

  return {
    setSection(index: number) {
      const next = Math.max(0, Math.min(CAMERAS.length - 1, index));
      if (next === current) return;
      previous = current;
      current = next;
      fade = 0;
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      scene.traverse((o) => {
        const mesh = o as THREE.Mesh;
        mesh.geometry?.dispose?.();
        const mat = mesh.material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose?.());
        else mat?.dispose?.();
      });
      renderer.dispose();
    },
  };
}
