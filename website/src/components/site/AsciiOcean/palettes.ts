// Adapted from stackedlol/ascii-sharks (MIT); see LICENSE in this folder.
/**
 * The ocean's two looks.
 *
 * night is the reference's: black water, pale fish, grey sharks shaded by the light and a cyan wake. The dot
 * classes, the heat where a fish is eaten and the hides' shimmer move onto the logo's gold (#fdb515) and blue
 * (#30a2ff).
 *
 * paper is for the light theme: the homepage's cream ground (#fcfbf8 at the top to the paper #f6f1e7), ink fish,
 * and sharks printed in ink like the reference's README art: the glyphs carry it, dense and dark in shadow, sparse
 * and lighter on lit skin, over a wash that runs from grey in shadow to almost the paper on the lit back. A blue
 * wake, gold heat, and gold and blue shimmer.
 *
 * Keep each ground in step with styles.module.css, which paints it before the first frame.
 */
import { clamp } from './math'

export type PaletteName = 'night' | 'paper'

export type Rgb = readonly [number, number, number]

export interface Palette {
  /** Vertical gradient stops for the ground, top (0) to bottom (1). */
  ground: readonly (readonly [number, string])[]
  grid: string
  /** Dot colour classes: blue, gold, deep blue, pale; the last entry is the wake. */
  dots: readonly Rgb[]
  /** Fish dashes, drawn on the reference's brightness ladder. */
  fish: Rgb
  /** Where a fish was eaten: hot, then fading. */
  heatHot: Rgb
  heatWarm: Rgb
  /** The shimmer: seven, picked per cell like the reference's. */
  accents: readonly string[]
  /** A skin block for the reference's grey level (0–255, lit skin high) under a species tint. */
  skin: (level: number, tint: Rgb) => Rgb
  /** An edge glyph (no block) for the reference's grey g, which rises with coverage. */
  edge: (g: number, tint: Rgb) => Rgb
  /** The glyphs of tiger bars and zebra spots. */
  mark: (tint: Rgb) => Rgb
  /** Glyph ink on lit and on shaded skin, the white glint, the eye. */
  inkLit: string
  inkShade: string
  glint: string
  eye: string
  /** Alpha of what's drawn inside a quiet rect: at most 0.2, so the copy keeps its contrast. */
  quiet: number
}

const channel = (v: number) => clamp(Math.round(v), 0, 255)

const lerp = (a: Rgb, b: Rgb, t: number, [r, g, bl]: Rgb): Rgb => [
  channel((a[0] + (b[0] - a[0]) * t) * r),
  channel((a[1] + (b[1] - a[1]) * t) * g),
  channel((a[2] + (b[2] - a[2]) * t) * bl),
]

// Paper keeps the reference's lighting: the wash is lightest where night's grey is brightest. Edge glyphs fade
// from soft grey to ink as coverage rises.
const INK: Rgb = [22, 24, 29]
const WASH_SHADOW: Rgb = [92, 98, 110]
const WASH_LIT: Rgb = [238, 236, 230]
const EDGE_FAINT: Rgb = [192, 192, 190]
const EDGE_FULL: Rgb = [58, 62, 72]

export const PALETTES: Record<PaletteName, Palette> = {
  night: {
    ground: [[0, '#000000'], [1, '#000000']],
    grid: 'rgba(255,255,255,0.055)',
    dots: [[48, 162, 255], [253, 181, 21], [8, 118, 201], [190, 212, 255], [110, 230, 255]],
    fish: [230, 238, 255],
    heatHot: [253, 181, 21],
    heatWarm: [236, 118, 24],
    accents: ['#fdb515', '#30a2ff', '#ffd36e', '#75c5ff', '#e8960f', '#2a7fe0', '#6ee6ff'],
    skin: (level, [r, g, b]) => [channel(level * r), channel(level * g), channel(level * b + 4)],
    edge: (g, [r, gg, b]) => [channel(g * r), channel(g * gg), channel(g * b + 6)],
    mark: ([r, g, b]) => [channel(70 * r), channel(70 * g), channel(70 * b)],
    inkLit: '#0e0f13',
    inkShade: '#1b1d24',
    glint: '#ffffff',
    eye: '#9aa0ad',
    quiet: 0.16,
  },
  paper: {
    ground: [[0, '#fcfbf8'], [0.5, '#faf8f3'], [0.82, '#f8f4ec'], [1, '#f6f1e7']],
    grid: 'rgba(22,24,29,0.05)',
    dots: [[48, 162, 255], [214, 148, 0], [8, 118, 201], [120, 126, 138], [22, 138, 236]],
    fish: [22, 24, 29],
    heatHot: [232, 160, 0],
    heatWarm: [208, 112, 12],
    accents: ['#f2a900', '#1f8fff', '#d48b00', '#0b6fc6', '#ffc23d', '#58b6ff', '#b97a00'],
    skin: (level, tint) => lerp(WASH_SHADOW, WASH_LIT, clamp(level / 226, 0, 1), tint),
    edge: (g, tint) => lerp(EDGE_FAINT, EDGE_FULL, clamp((g - 50) / 100, 0, 1), tint),
    mark: tint => lerp(INK, WASH_SHADOW, 0.4, tint),
    inkLit: '#3b404b',
    inkShade: '#16181d',
    glint: '#ffffff',
    eye: '#16181d',
    // Ink under the copy darkens the cream more than light does the black: at 0.08 even the muted 13.5 px film
    // link keeps 4.5:1 over a solid ink glyph.
    quiet: 0.08,
  },
}

export const rgb = ([r, g, b]: Rgb) => `rgb(${r},${g},${b})`

export const rgba = ([r, g, b]: Rgb, a: number) => `rgba(${r},${g},${b},${+a.toFixed(3)})`
