// Adapted from stackedlol/ascii-sharks (MIT); see LICENSE in this folder.
export const TAU = Math.PI * 2

export const clamp = (v: number, lo: number, hi: number) =>
  v < lo ? lo : v > hi ? hi : v

export const rand = (lo: number, hi: number) => lo + Math.random() * (hi - lo)

export const randInt = (lo: number, hi: number) =>
  Math.floor(rand(lo, hi + 1))

export function smoothstep(e0: number, e1: number, x: number) {
  const t = clamp((x - e0) / (e1 - e0), 0, 1)
  return t * t * (3 - 2 * t)
}

/** Shortest signed angle from a to b, in (-PI, PI]. */
export function angleDiff(a: number, b: number) {
  let d = (b - a) % TAU
  if (d > Math.PI) d -= TAU
  if (d < -Math.PI) d += TAU
  return d
}

/** Cheap integer hash → [0, 1). Deterministic, so textures stay stable. */
export function hash3(x: number, y: number, z: number) {
  let h
    = Math.imul(x | 0, 374761393)
      ^ Math.imul(y | 0, 668265263)
      ^ Math.imul(z | 0, 1442695041)
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  h ^= h >>> 16
  return (h >>> 0) / 4294967296
}

const K1 = 1 / 260
const K2 = 1 / 310
const K3 = 1 / 190
const CURRENT_SCALE = 3400

/**
 * Divergence-free ocean current: the curl of a slowly drifting stream
 * function ψ(x, y, t). Because it has no sources or sinks, it moves the
 * schools around without bunching them up in one corner.
 */
export function current(x: number, y: number, t: number, out: { x: number, y: number }) {
  const a = x * K1 + t * 0.07
  const b = y * K2 - t * 0.05
  const c = (x + y) * K3 + t * 0.11
  // u = ∂ψ/∂y, v = -∂ψ/∂x with ψ = sin(a)cos(b) + 0.5 sin(c)
  out.x = (-K2 * Math.sin(a) * Math.sin(b) + 0.5 * K3 * Math.cos(c)) * CURRENT_SCALE
  out.y = (-K1 * Math.cos(a) * Math.cos(b) - 0.5 * K3 * Math.cos(c)) * CURRENT_SCALE
  return out
}
