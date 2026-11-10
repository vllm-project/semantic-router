/* Shared by the homepage's canvases (the block field, the section bands, the mosaic rules): Duo's seeded random
 * numbers and smoothstep, and colours read from the duo-home CSS custom properties, so each canvas follows the
 * colour mode. */

export function rng(seed: number): () => number {
  let s = seed
  return () => {
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function smooth(a: number, b: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

// A custom property holding "r g b", as "r, g, b" for rgba().
export function rgbVar(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return (value || fallback).split(/[\s,]+/).join(', ')
}

export function numberVar(name: string, fallback: number): number {
  const value = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name))
  return Number.isFinite(value) ? value : fallback
}

// Calls back when the colour mode changes; returns the disconnect.
export function onThemeChange(callback: () => void): () => void {
  const observer = new MutationObserver(callback)
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
  return () => observer.disconnect()
}

export function reducedMotion(): MediaQueryList {
  return window.matchMedia('(prefers-reduced-motion: reduce)')
}
