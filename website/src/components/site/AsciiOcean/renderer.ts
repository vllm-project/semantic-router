// Adapted from stackedlol/ascii-sharks (MIT); see LICENSE in this folder.
/**
 * Draws the World as a grid of ASCII cells:
 *   1. a cell grid on the palette's ground
 *   2. a wake field that diffuses where things swim, lighting the dots
 *   3. fish as minimal ASCII dashes, one per cell, glinting on hard turns
 *   4. sharks rasterised cell-by-cell with supersampled edges: blocks
 *      shaded by the light, dense glyphs, and shimmering accent colours
 *   5. a veil of ground over the quiet rects, so the copy on top stays readable
 *
 * Each glyph is drawn once per colour into an atlas and stamped with drawImage, a fraction of the cost of a
 * fillText per cell; colours are quantised in steps too small to see, which keeps the atlas small. A shark only
 * samples the cells within reach of its spine.
 */
import { clamp, hash3, smoothstep } from './math'
import { rgb, rgba, type Palette } from './palettes'
import { SHARK_NODES, type Shark, type SharkKind, type World } from './world'

const ALPHA_LEVELS = 6
const FISH_ALPHAS = [0.28, 0.42, 0.58, 0.74, 0.9]

// Dot colour classes come from the palette: blue, gold, deep blue, pale; the last entry is the wake.
const DOT_CLASSES = 5
const DOT_BASE_ALPHA = [1, 1, 1, 1, 1]
const WAKE = DOT_CLASSES - 1

// Sparse glyphs on bright skin, dense glyphs in shadow, so the texture
// reinforces the shading instead of fighting it.
const SKIN_RAMP = codes(':;t%SX8#@')
const FIN_GLYPHS = codes('%;tS:')
const RIDGE_GLYPHS = codes('@8%')
const EDGE_HEAVY = codes(':;')
const EDGE_LIGHT = codes('.\',`')
const EYE_GLYPH = codes('o')[0]
const FISH_GLYPH = codes('-')[0]
const HEAT_HOT_GLYPH = codes('*')[0]
const HEAT_GLYPH = codes('·')[0]

const FLAG_BODY = 1
const FLAG_EDGE = 2
const FLAG_FIN = 3
const FLAG_RIDGE = 4
const FLAG_EYE = 5
/** Dark skin markings: tiger bars, zebra-shark spots. */
const FLAG_MARK = 6
const MARK_GLYPHS = codes('#%@')

// Sub-cell sample offsets for anti-aliased silhouettes.
const SUBSAMPLES = [0.25, 0.3, 0.75, 0.3, 0.25, 0.72, 0.75, 0.72]

/** Skin blocks are quantised to steps of 2 grey levels, edge glyphs likewise, heat alpha to 1/20. */
const LEVELS = 128
const EDGE_LEVELS = 96
const HEAT_LEVELS = 20
/** What shows through the quiet rects' two feathering rings of cells, after the rects' own palette.quiet. */
const RING_ALPHA = [0.45, 0.75]

function codes(glyphs: string) {
  return Array.from(glyphs, g => g.charCodeAt(0))
}

/** Classic top-down half-width at s ∈ [0, 1] (snout → tail), as a fraction of length. */
function classicHalfWidth(s: number) {
  if (s < 0.13) return 0.09 * Math.pow(Math.sin(((s / 0.13) * Math.PI) / 2), 0.95)
  if (s < 0.3) return 0.09 + 0.016 * smoothstep(0.13, 0.3, s)
  if (s < 0.84) return 0.019 + 0.087 * Math.pow(1 - (s - 0.3) / 0.54, 1.25)
  const k = (s - 0.84) / 0.16
  return 0.019 * (1 - k) + 0.012 * Math.sin(k * Math.PI)
}

/** Body outline and fin layout per species. All values are fractions of length. */
interface SharkShape {
  halfWidth: (s: number) => number
  /** Pectoral fin: leading root, trailing root, tip position along the body, and reach. */
  pectoral: [number, number, number, number]
  /** Notched (falcate) trailing edge on the pectorals. */
  falcate: boolean
  /** Pelvic fins: leading root, trailing root, tip, reach. */
  pelvic: [number, number, number, number]
  /** Forked caudal fin reach, or 0 when the tail is a ribbon (zebra shark). */
  caudal: number
  dorsal: [number, number]
  secondDorsal: [number, number]
  /**
   * Hammerheads' cephalofoil: half-span, how far its tips sweep back, its depth, and the
   * notch in the middle of the front edge. All fractions of length; absent for other sharks.
   */
  head?: { span: number, sweep: number, depth: number, notch: number }
  /** null when the eyes sit somewhere else (hammerhead: on the cephalofoil tips). */
  eye: [number, number] | null
  gills: [number, number]
  /** Skin colour multipliers (r, g, b). */
  tint: [number, number, number]
}

const SHAPES: Record<SharkKind, SharkShape> = {
  classic: {
    halfWidth: classicHalfWidth,
    pectoral: [0.15, 0.43, 0.55, 0.26],
    falcate: true,
    pelvic: [0.57, 0.65, 0.69, 0.085],
    caudal: 0.085,
    dorsal: [0.33, 0.47],
    secondDorsal: [0.71, 0.76],
    eye: [0.1, 0.14],
    gills: [0.19, 0.29],
    tint: [0.92, 0.97, 1.05],
  },
  // Slim neck under a wide, swept-back cephalofoil (drawn in sampleShark),
  // tall first dorsal, long upper tail lobe. Bronze-olive grey.
  hammerhead: {
    halfWidth: (s) => {
      if (s < 0.06) return 0.045
      if (s < 0.3) return 0.045 + 0.061 * smoothstep(0.06, 0.3, s)
      return classicHalfWidth(s) * 0.92
    },
    head: { span: 0.17, sweep: 0.035, depth: 0.09, notch: 0 },
    pectoral: [0.17, 0.4, 0.5, 0.22],
    falcate: true,
    pelvic: [0.56, 0.63, 0.67, 0.075],
    caudal: 0.11,
    dorsal: [0.28, 0.44],
    secondDorsal: [0.7, 0.74],
    eye: null,
    gills: [0.16, 0.26],
    tint: [0.96, 0.98, 0.9],
  },
  // Blunt, square snout on a heavy body; barred back. Warm grey-brown.
  tiger: {
    halfWidth: (s) => {
      if (s < 0.13) return 0.1 * Math.pow(Math.sin(((s / 0.13) * Math.PI) / 2), 0.45)
      if (s < 0.3) return 0.1 + 0.014 * smoothstep(0.13, 0.3, s)
      if (s < 0.84) return 0.019 + 0.095 * Math.pow(1 - (s - 0.3) / 0.54, 1.2)
      const k = (s - 0.84) / 0.16
      return 0.019 * (1 - k) + 0.012 * Math.sin(k * Math.PI)
    },
    pectoral: [0.15, 0.42, 0.53, 0.25],
    falcate: true,
    pelvic: [0.57, 0.65, 0.69, 0.085],
    caudal: 0.095,
    dorsal: [0.33, 0.46],
    secondDorsal: [0.71, 0.75],
    eye: [0.08, 0.12],
    gills: [0.19, 0.28],
    tint: [1.02, 0.95, 0.84],
  },
  // Short rounded head, body done by half-length, then a long ribbon of a
  // tail fin. Broad rounded pectorals, spotted sandy skin.
  zebra: {
    halfWidth: (s) => {
      if (s < 0.1) return 0.075 * Math.pow(Math.sin(((s / 0.1) * Math.PI) / 2), 0.7)
      if (s < 0.28) return 0.075 + 0.025 * smoothstep(0.1, 0.28, s)
      if (s < 0.55) return 0.022 + 0.078 * Math.pow(1 - (s - 0.28) / 0.27, 1.1)
      const k = (s - 0.55) / 0.45
      return 0.022 * (1 - k * k) + 0.012 * Math.sin(k * Math.PI)
    },
    pectoral: [0.12, 0.3, 0.36, 0.19],
    falcate: false,
    pelvic: [0.42, 0.48, 0.51, 0.065],
    caudal: 0,
    dorsal: [0.36, 0.44],
    secondDorsal: [0.49, 0.53],
    eye: [0.06, 0.09],
    gills: [0.13, 0.2],
    tint: [1.08, 0.97, 0.74],
  },
  // Great hammerhead: a broader, nearly straight-edged cephalofoil with a notch at the centre,
  // a towering sickle first dorsal (drawn long and bright from above), and a cooler slate skin.
  greathammer: {
    halfWidth: (s) => {
      if (s < 0.05) return 0.04
      if (s < 0.3) return 0.04 + 0.07 * smoothstep(0.05, 0.3, s)
      return classicHalfWidth(s) * 0.95
    },
    head: { span: 0.21, sweep: 0.006, depth: 0.085, notch: 0.014 },
    pectoral: [0.16, 0.38, 0.49, 0.24],
    falcate: true,
    pelvic: [0.55, 0.62, 0.67, 0.085],
    caudal: 0.13,
    dorsal: [0.24, 0.45],
    secondDorsal: [0.68, 0.74],
    eye: null,
    gills: [0.15, 0.25],
    tint: [0.86, 0.93, 1.02],
  },
  // Shortfin mako: a long conical snout, a torpedo body that narrows hard to the tail stalk,
  // and a tall crescent tail. Metallic indigo back.
  mako: {
    halfWidth: (s) => {
      if (s < 0.17) return 0.088 * Math.pow(Math.sin(((s / 0.17) * Math.PI) / 2), 1.5)
      if (s < 0.32) return 0.088 + 0.012 * smoothstep(0.17, 0.32, s)
      if (s < 0.86) return 0.014 + 0.086 * Math.pow(1 - (s - 0.32) / 0.54, 1.45)
      const k = (s - 0.86) / 0.14
      return 0.014 * (1 - k) + 0.01 * Math.sin(k * Math.PI)
    },
    pectoral: [0.18, 0.4, 0.5, 0.21],
    falcate: true,
    pelvic: [0.58, 0.64, 0.68, 0.06],
    caudal: 0.13,
    dorsal: [0.34, 0.45],
    secondDorsal: [0.76, 0.79],
    eye: [0.11, 0.15],
    gills: [0.2, 0.3],
    tint: [0.78, 0.9, 1.2],
  },
  // Thresher: a compact body over the front half, then the upper tail lobe as a long whip
  // that's as long as the shark itself. Big eyes, long sickle pectorals, bronze-violet skin.
  thresher: {
    halfWidth: (s) => {
      if (s < 0.08) return 0.07 * Math.pow(Math.sin(((s / 0.08) * Math.PI) / 2), 0.75)
      if (s < 0.2) return 0.07 + 0.02 * smoothstep(0.08, 0.2, s)
      if (s < 0.48) return 0.012 + 0.078 * Math.pow(1 - (s - 0.2) / 0.28, 1.3)
      // The whip: thin, with a slight sickle swell before it tapers out.
      const k = (s - 0.48) / 0.52
      return 0.012 * (1 - k) + 0.006 * Math.sin(k * Math.PI * 0.9) + 0.002
    },
    pectoral: [0.12, 0.27, 0.36, 0.2],
    falcate: true,
    pelvic: [0.37, 0.41, 0.44, 0.045],
    caudal: 0,
    dorsal: [0.24, 0.32],
    secondDorsal: [0.45, 0.47],
    eye: [0.05, 0.1],
    gills: [0.11, 0.18],
    tint: [1.0, 0.9, 1.08],
  },
  // Nurse shark: a blunt, broad head, a heavy flattened body with both dorsals set far back,
  // rounded paddle pectorals, and a long low tail with no lower lobe. Tawny brown.
  nurse: {
    halfWidth: (s) => {
      if (s < 0.08) return 0.085 * Math.pow(Math.sin(((s / 0.08) * Math.PI) / 2), 0.4)
      if (s < 0.3) return 0.085 + 0.02 * smoothstep(0.08, 0.3, s)
      if (s < 0.72) return 0.022 + 0.083 * Math.pow(1 - (s - 0.3) / 0.42, 1.1)
      const k = (s - 0.72) / 0.28
      return 0.022 * (1 - k * k) + 0.014 * Math.sin(k * Math.PI)
    },
    pectoral: [0.16, 0.34, 0.4, 0.17],
    falcate: false,
    pelvic: [0.45, 0.52, 0.55, 0.075],
    caudal: 0,
    dorsal: [0.5, 0.58],
    secondDorsal: [0.63, 0.69],
    eye: [0.05, 0.075],
    gills: [0.14, 0.22],
    tint: [1.1, 0.96, 0.76],
  },
}

const KINDS = Object.keys(SHAPES) as SharkKind[]

function inTriangle(
  px: number, py: number,
  ax: number, ay: number, bx: number, by: number, cx: number, cy: number,
) {
  const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
  const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
  const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
  const neg = d1 < 0 || d2 < 0 || d3 < 0
  const pos = d1 > 0 || d2 > 0 || d3 > 0
  return !(neg && pos)
}

const WIDTH_STEPS = 512
const REACH_BINS = 64

interface KindTables {
  /** halfWidth sampled along the body: a lookup instead of a pow and a sin per sample. */
  width: Float32Array
  /** How far the silhouette reaches from the spine in each 1/64 of it, in lengths, pectorals aside (they flex). */
  reach: Float32Array
}

const kindTables = new Map<SharkKind, KindTables>()

function tablesFor(kind: SharkKind): KindTables {
  let tables = kindTables.get(kind)
  if (tables) return tables
  const shape = SHAPES[kind]
  const width = new Float32Array(WIDTH_STEPS + 1)
  for (let i = 0; i <= WIDTH_STEPS; i++) width[i] = shape.halfWidth(i / WIDTH_STEPS)
  const reach = new Float32Array(REACH_BINS)
  const { head, pelvic, caudal } = shape
  for (let b = 0; b < REACH_BINS; b++) {
    const s0 = b / REACH_BINS
    const s1 = (b + 1) / REACH_BINS
    let r = 0
    for (let k = 0; k <= 8; k++) r = Math.max(r, shape.halfWidth(s0 + ((s1 - s0) * k) / 8))
    if (head && s0 < head.sweep + head.depth + head.notch) r = Math.max(r, head.span)
    if (s1 > pelvic[0] && s0 < pelvic[2]) r = Math.max(r, pelvic[3])
    if (caudal > 0 && s1 > 0.86) r = Math.max(r, caudal)
    reach[b] = r * 1.05 + 0.002
  }
  tables = { width, reach }
  kindTables.set(kind, tables)
  return tables
}

/** Visible cell grid, snapped to device pixels. */
function paintGrid(g: CanvasRenderingContext2D, W: number, H: number, cw: number, ch: number, dpr: number, stroke: string) {
  g.setTransform(1, 0, 0, 1, 0, 0)
  g.strokeStyle = stroke
  g.lineWidth = 1
  g.beginPath()
  for (let c = 0; c <= Math.ceil(W / cw); c++) {
    const x = Math.round(c * cw * dpr) + 0.5
    g.moveTo(x, 0)
    g.lineTo(x, H * dpr)
  }
  for (let r = 0; r <= Math.ceil(H / ch); r++) {
    const y = Math.round(r * ch * dpr) + 0.5
    g.moveTo(0, y)
    g.lineTo(W * dpr, y)
  }
  g.stroke()
}

const ATLAS_ROW = 32
const ATLAS_MAX = 2048

/** Glyph sprites, one per glyph and colour, drawn on first use into an offscreen canvas. */
class GlyphAtlas {
  canvas: HTMLCanvasElement
  /** Slot size and the margin around the cell, in device px. */
  w = 1
  h = 1
  pad = 0
  x = new Int32Array(0)
  y = new Int32Array(0)
  private g: CanvasRenderingContext2D
  private capacity = 0
  private used = 0
  private slots = new Map<number, number>()
  private colors: string[] = []
  private colorIds = new Map<string, number>()
  private font = ''
  private ox = 0
  private oy = 0

  constructor() {
    this.canvas = document.createElement('canvas')
    this.g = this.context(this.canvas)
  }

  /** Lays the slots out for a cell size, DPR and font, dropping every sprite. */
  reset(cw: number, ch: number, dpr: number, family: string) {
    this.pad = Math.ceil(2 * dpr)
    this.w = Math.ceil(cw * dpr) + this.pad * 2
    this.h = Math.ceil(ch * dpr) + this.pad * 2
    this.ox = this.pad + (cw * dpr) / 2
    this.oy = this.pad + (ch / 2 + 1) * dpr
    this.font = `600 ${Math.round(ch * 0.74) * dpr}px ${family}`
    this.layout(256)
  }

  /** Drops every sprite, so they're drawn again (with a font that has since loaded, say). */
  clear() {
    this.layout(this.capacity || 256)
  }

  /** Frees the canvas. */
  release() {
    this.canvas.width = 0
    this.canvas.height = 0
  }

  /** A stable id for a CSS colour. */
  color(css: string) {
    let id = this.colorIds.get(css)
    if (id === undefined) {
      id = this.colors.length
      this.colors.push(css)
      this.colorIds.set(css, id)
    }
    return id
  }

  /** The slot of a glyph (char code below 256) in a colour, drawn now if it's new. */
  sprite(code: number, color: number) {
    const key = color * 256 + code
    const known = this.slots.get(key)
    if (known !== undefined) return known
    // Slots handed out this frame are drawn later in it, so a full atlas grows and keeps its sprites.
    if (this.used === this.capacity && !this.grow()) return this.used - 1
    const slot = this.used++
    const g = this.g
    g.font = this.font
    g.textAlign = 'center'
    g.textBaseline = 'middle'
    g.fillStyle = this.colors[color]
    g.fillText(String.fromCharCode(code), this.x[slot] + this.ox, this.y[slot] + this.oy)
    this.slots.set(key, slot)
    return slot
  }

  private context(canvas: HTMLCanvasElement) {
    const g = canvas.getContext('2d')
    if (!g) throw new Error('2D canvas unavailable')
    return g
  }

  private slotsFor(capacity: number) {
    this.capacity = capacity
    this.x = new Int32Array(capacity)
    this.y = new Int32Array(capacity)
    for (let s = 0; s < capacity; s++) {
      this.x[s] = (s % ATLAS_ROW) * this.w
      this.y[s] = Math.floor(s / ATLAS_ROW) * this.h
    }
  }

  private layout(capacity: number) {
    this.slotsFor(capacity)
    this.used = 0
    this.slots.clear()
    this.canvas.width = ATLAS_ROW * this.w
    this.canvas.height = Math.ceil(capacity / ATLAS_ROW) * this.h
  }

  private grow() {
    if (this.capacity >= ATLAS_MAX) return false
    const next = document.createElement('canvas')
    next.width = this.canvas.width
    next.height = this.canvas.height * 2
    const g = this.context(next)
    g.drawImage(this.canvas, 0, 0)
    this.release()
    this.canvas = next
    this.g = g
    this.slotsFor(this.capacity * 2)
    return true
  }
}

export class AsciiRenderer {
  private ctx: CanvasRenderingContext2D
  /** Ground and grid, painted once per size and palette. */
  private bg: HTMLCanvasElement
  private atlas: GlyphAtlas
  private palette: Palette
  private dpr = 1
  private width = 0
  private height = 0
  private cw = 12
  private ch = 18
  private cols = 0
  private rows = 0
  private fontFamily = 'ui-monospace, Menlo, monospace'
  /** Device-px origin of each column and row. */
  private colX = new Int32Array(0)
  private rowY = new Int32Array(0)

  private dotKind = new Uint8Array(0) // 0 none, 1 blue, 2 gold, 3 deep blue, 4 pale
  private dotPhase = new Float32Array(0)
  private dotFreq = new Float32Array(0)
  private energy = new Float32Array(0)
  private energyTmp = new Float32Array(0)
  private heat = new Float32Array(0)
  private bucketOf = new Uint8Array(0)
  private bucketOrder = new Int32Array(0)
  private bucketCounts = new Int32Array(DOT_CLASSES * ALPHA_LEVELS + 1)
  private bucketStarts = new Int32Array(DOT_CLASSES * ALPHA_LEVELS + 1)
  private bucketCursor = new Int32Array(DOT_CLASSES * ALPHA_LEVELS + 1)
  private dotStyles: string[] = []

  /** Cells a shark, a shark's shadow or a fish holds, by frame stamp, so nothing is cleared per frame. */
  private frameId = 0
  private sharkStamp = new Uint32Array(0)
  private shadowStamp = new Uint32Array(0)
  private fishStamp = new Uint32Array(0)
  private visit = new Uint32Array(0)
  private visitId = 0
  /** This frame's shark cells, and per cell its block fill (-1 for an edge glyph) and glyph sprite. */
  private sharkCells = new Int32Array(0)
  private sharkCellCount = 0
  private cellBlock = new Int16Array(0)
  private cellSprite = new Int32Array(0)
  private blockOrder = new Int32Array(0)
  private blockCounts = new Int32Array(KINDS.length * LEVELS)
  private blockCursor = new Int32Array(KINDS.length * LEVELS)
  private blocksUsed: number[] = []
  private sharkFlag = new Uint8Array(0)
  private sharkShade = new Float32Array(0)
  private sharkKey = new Uint32Array(0)
  private sharkDepth = new Float32Array(0)
  private sharkKind = new Uint8Array(0)
  private spine = new Float32Array(SHARK_NODES * 2)
  private arc = new Float32Array(SHARK_NODES)
  private arcTotal = 1
  /** Per render-spine segment: start, direction, length and 1/length², rebuilt per shark. */
  private segAx = new Float32Array(SHARK_NODES)
  private segAy = new Float32Array(SHARK_NODES)
  private segSx = new Float32Array(SHARK_NODES)
  private segSy = new Float32Array(SHARK_NODES)
  private segLen = new Float32Array(SHARK_NODES)
  private segInv = new Float32Array(SHARK_NODES)
  private shape = SHAPES.classic
  private widths: Float32Array = new Float32Array(WIDTH_STEPS + 1)
  private reachRaw = new Float32Array(REACH_BINS)
  private reach = new Float32Array(REACH_BINS)
  private sharkId = 0
  private finTipLeft = 0
  private finTipRight = 0
  // Scratch outputs of nearest() and sampleShark(), to avoid allocating per sample.
  private hitFlag = 0
  private hitShade = 0
  private hitS = 0
  private hitV = 0
  private hitDist = 0
  private hitK = 0
  private hitU = 0

  /** Fish cells bucketed by brightness, so each bucket is one fill. */
  private fishBuckets: number[][] = FISH_ALPHAS.map(() => [])
  /** The fish glyph's ink box from its cell's corner, in device px: a dash is a bar, so it's drawn as one. */
  private dash = { x: 0, y: 0, w: 0, h: 0 }

  // The palette as atlas colour ids and fills, and the skin fills per species and grey level.
  private fishStyles: string[] = []
  private heatHot: number[] = []
  private heatWarm: number[] = []
  private accents: number[] = []
  private inkLit = 0
  private inkShade = 0
  private glint = 0
  private eye = 0
  private markColors: number[] = []
  private edgeColors: number[][] = []
  private blockStyles: string[] = []

  /** The veil over the quiet rects: [x, y, w, h] runs in device px, the rects' cells first, then each ring. */
  private quiet: Float32Array = new Float32Array(0)
  private veil = new Int32Array(0)
  private veilEnds = [0, 0, 0]

  constructor(private canvas: HTMLCanvasElement, palette: Palette) {
    const ctx = canvas.getContext('2d', { alpha: false })
    if (!ctx) throw new Error('2D canvas unavailable')
    this.ctx = ctx
    this.bg = document.createElement('canvas')
    this.atlas = new GlyphAtlas()
    this.palette = palette
    this.buildStyles()
  }

  get cellWidth() {
    return this.cw
  }

  get cellHeight() {
    return this.ch
  }

  setFontFamily(family: string) {
    if (!family || family === this.fontFamily) return
    this.fontFamily = family
    if (!this.cols) return
    this.atlas.reset(this.cw, this.ch, this.dpr, family)
    this.measureDash()
  }

  /** Redraws every glyph sprite: call once the web font has loaded. */
  refreshGlyphs() {
    this.atlas.clear()
    if (this.cols) this.measureDash()
  }

  setPalette(palette: Palette) {
    if (palette === this.palette) return
    this.palette = palette
    this.buildStyles()
    this.atlas.clear()
    this.paintBackground()
  }

  /** The hero copy's rects in canvas px, [x0, y0, x1, y1] each: drawn under a veil of ground. */
  setQuiet(rects: Float32Array) {
    this.quiet = rects
    this.buildVeil()
  }

  /** Frees the offscreen canvases. */
  destroy() {
    this.atlas.release()
    this.bg.width = 0
    this.bg.height = 0
  }

  resize(width: number, height: number, dpr: number) {
    const prevCols = this.cols
    const prevRows = this.rows
    const prevCw = this.cw
    const prevDpr = this.dpr
    this.width = width
    this.height = height
    this.dpr = dpr
    this.cw = clamp(Math.round(width / 124), 8, 13)
    this.ch = Math.round(this.cw * 1.45)
    this.cols = Math.ceil(width / this.cw)
    this.rows = Math.ceil(height / this.ch)

    this.canvas.width = Math.round(width * dpr)
    this.canvas.height = Math.round(height * dpr)
    this.bg.width = this.canvas.width
    this.bg.height = this.canvas.height

    const n = this.cols * this.rows
    if (this.cols !== prevCols || this.rows !== prevRows) {
      this.dotKind = new Uint8Array(n)
      this.dotPhase = new Float32Array(n)
      this.dotFreq = new Float32Array(n)
      this.energy = new Float32Array(n)
      this.energyTmp = new Float32Array(n)
      this.heat = new Float32Array(n)
      this.bucketOf = new Uint8Array(n)
      this.bucketOrder = new Int32Array(n)
      this.sharkStamp = new Uint32Array(n)
      this.shadowStamp = new Uint32Array(n)
      this.fishStamp = new Uint32Array(n)
      this.visit = new Uint32Array(n)
      this.sharkCells = new Int32Array(n)
      this.cellBlock = new Int16Array(n)
      this.cellSprite = new Int32Array(n)
      this.blockOrder = new Int32Array(n)
      this.sharkFlag = new Uint8Array(n)
      this.sharkShade = new Float32Array(n)
      this.sharkKey = new Uint32Array(n)
      this.sharkDepth = new Float32Array(n)
      this.sharkKind = new Uint8Array(n)
    }

    for (let i = 0; i < n; i++) {
      const col = i % this.cols
      const row = (i / this.cols) | 0
      // No ambient dots: the grid stays black. Only the cyan wake lights cells.
      this.dotKind[i] = 0
      this.dotPhase[i] = hash3(col, row, 23) * Math.PI * 2
      this.dotFreq[i] = 0.3 + hash3(col, row, 37) * 1.1
    }

    this.colX = new Int32Array(this.cols)
    this.rowY = new Int32Array(this.rows)
    for (let c = 0; c < this.cols; c++) this.colX[c] = Math.round(c * this.cw * dpr)
    for (let r = 0; r < this.rows; r++) this.rowY[r] = Math.round(r * this.ch * dpr)
    if (this.cw !== prevCw || dpr !== prevDpr || !prevCols) this.atlas.reset(this.cw, this.ch, dpr, this.fontFamily)
    this.measureDash()
    this.paintBackground()
    this.buildVeil()
  }

  private paintBackground() {
    const g = this.bg.getContext('2d')
    if (!g || this.cols === 0) return
    const { width: W, height: H, dpr, cw, ch, palette } = this
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    const stops = palette.ground
    if (stops.every(([, color]) => color === stops[0][1])) {
      g.fillStyle = stops[0][1]
    }
    else {
      const gradient = g.createLinearGradient(0, 0, 0, H)
      for (const [at, color] of stops) gradient.addColorStop(at, color)
      g.fillStyle = gradient
    }
    g.fillRect(0, 0, W, H)
    paintGrid(g, W, H, cw, ch, dpr, palette.grid)
  }

  /** Moves the wake and heat fields on by one sim step. */
  advance(world: World, dt: number) {
    if (this.cols === 0) return
    this.consumeEvents(world)
    this.depositWake(world)
    this.diffuse(dt)
  }

  /** Paints the bare water: ground and grid. */
  clear() {
    if (this.cols === 0) return
    const { ctx } = this
    ctx.globalAlpha = 1
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.drawImage(this.bg, 0, 0)
  }

  render(world: World) {
    const { ctx, dpr } = this
    if (this.cols === 0) return
    const t = world.time
    const frame = ++this.frameId

    this.rasterizeSharks(world.sharks, t, frame)
    this.markFish(world, frame)

    this.clear()
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    this.drawDots(t, frame)
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    this.drawHeat()
    this.drawFish()
    this.drawSharks(t)
    this.drawVeil()
  }

  private buildStyles() {
    const p = this.palette
    const atlas = this.atlas
    this.dotStyles = []
    for (let c = 0; c < DOT_CLASSES; c++) {
      for (let l = 0; l < ALPHA_LEVELS; l++) {
        this.dotStyles.push(rgba(p.dots[c], (DOT_BASE_ALPHA[c] * (l + 1)) / ALPHA_LEVELS))
      }
    }
    this.fishStyles = FISH_ALPHAS.map(a => rgba(p.fish, a))
    this.heatHot = []
    this.heatWarm = []
    for (let q = 0; q <= HEAT_LEVELS; q++) {
      this.heatHot.push(atlas.color(rgba(p.heatHot, q / HEAT_LEVELS)))
      this.heatWarm.push(atlas.color(rgba(p.heatWarm, q / HEAT_LEVELS)))
    }
    this.accents = p.accents.map(c => atlas.color(c))
    this.inkLit = atlas.color(p.inkLit)
    this.inkShade = atlas.color(p.inkShade)
    this.glint = atlas.color(p.glint)
    this.eye = atlas.color(p.eye)
    this.blockStyles = []
    this.edgeColors = []
    this.markColors = []
    for (const kind of KINDS) {
      const tint = SHAPES[kind].tint
      for (let q = 0; q < LEVELS; q++) this.blockStyles.push(rgb(p.skin(q * 2, tint)))
      const edges: number[] = []
      for (let q = 0; q < EDGE_LEVELS; q++) edges.push(atlas.color(rgb(p.edge(q * 2, tint))))
      this.edgeColors.push(edges)
      this.markColors.push(atlas.color(rgb(p.mark(tint))))
    }
  }

  /** Cells under a quiet rect get the strongest veil; two rings of cells around it feather the edge. */
  private buildVeil() {
    const { cols, rows, cw, ch, quiet } = this
    const strength = new Uint8Array(cols * rows)
    const pad = 2
    for (let q = 0; q < quiet.length; q += 4) {
      const c0 = Math.floor((quiet[q] - pad) / cw)
      const r0 = Math.floor((quiet[q + 1] - pad) / ch)
      const c1 = Math.floor((quiet[q + 2] + pad) / cw)
      const r1 = Math.floor((quiet[q + 3] + pad) / ch)
      for (let ring = 2; ring >= 0; ring--) {
        const s = 3 - ring
        for (let r = Math.max(0, r0 - ring); r <= Math.min(rows - 1, r1 + ring); r++) {
          for (let c = Math.max(0, c0 - ring); c <= Math.min(cols - 1, c1 + ring); c++) {
            const i = r * cols + c
            if (strength[i] < s) strength[i] = s
          }
        }
      }
    }
    // Runs of equal strength along each row, carried down the rows while they keep the same span.
    const { colX, rowY } = this
    const runs: number[][] = [[], [], []]
    let open = new Map<number, number>()
    for (let r = 0; r < rows; r++) {
      const y = rowY[r]
      const h = (r + 1 < rows ? rowY[r + 1] : this.canvas.height) - y
      const carried = new Map<number, number>()
      let c = 0
      while (c < cols) {
        const s = strength[r * cols + c]
        let e = c + 1
        while (e < cols && strength[r * cols + e] === s) e++
        if (s) {
          const key = (c * 4096 + e) * 4 + s
          const list = runs[3 - s]
          const at = open.get(key) ?? -1
          if (at >= 0) {
            list[at + 3] += h
            carried.set(key, at)
          }
          else {
            carried.set(key, list.length)
            list.push(colX[c], y, (e < cols ? colX[e] : this.canvas.width) - colX[c], h)
          }
        }
        c = e
      }
      open = carried
    }
    this.veil = new Int32Array([...runs[0], ...runs[1], ...runs[2]])
    this.veilEnds = [runs[0].length, runs[0].length + runs[1].length, this.veil.length]
  }

  // ── simulation → field ──────────────────────────────────────────────

  private cellIndex(x: number, y: number) {
    const c = Math.floor(x / this.cw)
    const r = Math.floor(y / this.ch)
    if (c < 0 || r < 0 || c >= this.cols || r >= this.rows) return -1
    return r * this.cols + c
  }

  private stamp(field: Float32Array, x: number, y: number, radius: number, amount: number) {
    const c0 = Math.floor(x / this.cw)
    const r0 = Math.floor(y / this.ch)
    for (let r = r0 - radius; r <= r0 + radius; r++) {
      for (let c = c0 - radius; c <= c0 + radius; c++) {
        if (c < 0 || r < 0 || c >= this.cols || r >= this.rows) continue
        const d = Math.hypot(c - c0, (r - r0) * 1.4)
        if (d > radius + 0.5) continue
        const i = r * this.cols + c
        field[i] = Math.min(1.6, field[i] + amount * (1 - d / (radius + 1)))
      }
    }
  }

  private consumeEvents(world: World) {
    for (const e of world.events) {
      if (e.kind === 'eat') {
        this.stamp(this.heat, e.x, e.y, 2, 1)
        this.stamp(this.energy, e.x, e.y, 3, 0.9)
      }
      else {
        this.stamp(this.energy, e.x, e.y, 6, 1.4)
      }
    }
    world.events.length = 0
  }

  private depositWake(world: World) {
    const { energy } = this
    for (const f of world.fish) {
      if (!f.alive) continue
      const i = this.cellIndex(f.x, f.y)
      if (i >= 0) energy[i] = Math.min(1.6, energy[i] + 0.012 + f.alarm * 0.03)
    }
    for (const sh of world.sharks) {
      // The tail sweeps the most water.
      for (let k = SHARK_NODES - 6; k < SHARK_NODES; k++) {
        const i = this.cellIndex(sh.nodes[k * 2], sh.nodes[k * 2 + 1])
        if (i >= 0) energy[i] = Math.min(1.6, energy[i] + 0.05 + sh.speed * 0.0012)
      }
    }
    if (world.pointer.active) {
      this.stamp(energy, world.pointer.x, world.pointer.y, 1, 0.08)
    }
  }

  /** Blur + decay so wakes spread out like ripples and fade. */
  private diffuse(dt: number) {
    const { cols, rows, energy, energyTmp: tmp, heat } = this
    const decay = Math.exp(-dt * 1.5)
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c
        const l = c > 0 ? energy[i - 1] : energy[i]
        const rt = c < cols - 1 ? energy[i + 1] : energy[i]
        const u = r > 0 ? energy[i - cols] : energy[i]
        const d = r < rows - 1 ? energy[i + cols] : energy[i]
        tmp[i] = (energy[i] * 4 + l + rt + u + d) * 0.125 * decay
      }
    }
    this.energy = tmp
    this.energyTmp = energy
    const heatDecay = Math.exp(-dt * 2.2)
    for (let i = 0; i < heat.length; i++) heat[i] *= heatDecay
  }

  // ── sharks ──────────────────────────────────────────────────────────

  /** The nearest render-spine point to (px, py) on segments k0 to k1 - 1: writes hitK, hitU (along it) and hitDist. */
  private nearest(px: number, py: number, k0: number, k1: number) {
    const { segAx, segAy, segSx, segSy, segInv } = this
    let best = Infinity
    let bestK = k0
    let bestU = 0
    for (let k = k0; k < k1; k++) {
      const dx = px - segAx[k]
      const dy = py - segAy[k]
      const sx = segSx[k]
      const sy = segSy[k]
      const u = clamp((dx * sx + dy * sy) * segInv[k], 0, 1)
      const qx = dx - sx * u
      const qy = dy - sy * u
      const d2 = qx * qx + qy * qy
      if (d2 < best) {
        best = d2
        bestK = k
        bestU = u
      }
    }
    this.hitK = bestK
    this.hitU = bestU
    this.hitDist = Math.sqrt(best)
  }

  /** Which side of the spine (px, py) is on, at the segment nearest() found: 1 or -1. */
  private side(px: number, py: number) {
    const k = this.hitK
    return this.segSx[k] * (py - this.segAy[k]) - this.segSy[k] * (px - this.segAx[k]) >= 0 ? 1 : -1
  }

  /** The body's half-width at s, in px. */
  private halfWidthAt(s: number, L: number) {
    const wi = s * WIDTH_STEPS
    const w0 = Math.min(WIDTH_STEPS - 1, wi | 0)
    return (this.widths[w0] + (this.widths[w0 + 1] - this.widths[w0]) * (wi - w0)) * L
  }

  /** Whether a point at distance a from the spine, at u = s·L along it and on side sign, is on the shark. */
  private covers(a: number, u: number, L: number, sign: number) {
    const shape = this.shape
    const head = shape.head
    if (head && u < L * (head.sweep + head.depth + head.notch)) {
      const r = a / (L * head.span)
      const front = L * head.sweep * r * r + L * head.notch * Math.max(0, 1 - r / 0.14)
      if (r <= 1 && u >= front && u <= front + L * head.depth * (1 - 0.2 * r * r * r * r)) return true
    }
    if (a <= this.halfWidthAt(u / L, L)) return true
    const tip = sign > 0 ? this.finTipRight : this.finTipLeft
    const [p0, p1, p2] = shape.pectoral
    if (
      inTriangle(u, a, L * p0, 0, L * p1, 0, L * p2, tip)
      && !(shape.falcate && inTriangle(u, a, L * p1, 0, L * (p1 + (p2 - p1) * 0.33), tip * 0.55, L * p2, tip))
    ) return true
    const [v0, v1, v2, vr] = shape.pelvic
    return inTriangle(u, a, L * v0, 0, L * v1, 0, L * v2, L * vr)
      || (shape.caudal > 0 && inTriangle(u, a, L * 0.86, 0, L * 0.95, 0, L, L * shape.caudal))
  }

  /** sampleShark() without the shading: whether a subsample is on the shark, its nearest segment in k0 to k1 - 1. */
  private inside(px: number, py: number, L: number, k0: number, k1: number) {
    this.nearest(px, py, k0, k1)
    const k = this.hitK
    const u = ((this.arc[k] + this.hitU * this.segLen[k]) / this.arcTotal) * L
    return this.covers(this.hitDist, u, L, this.side(px, py))
  }

  /**
   * Classify one point against the current shark (spine/arc already built).
   * Body coordinates: s along the spine (0 snout → 1 tail), v across it.
   * Writes hit* fields and returns whether the point is on the shark.
   */
  private sampleShark(px: number, py: number, L: number) {
    this.nearest(px, py, 0, SHARK_NODES - 1)
    const k = this.hitK
    const len = this.segLen[k]
    const bestSign = this.side(px, py)
    const bestNx = (-this.segSy[k] / len) * bestSign
    const bestNy = (this.segSx[k] / len) * bestSign

    const a = this.hitDist
    const s = (this.arc[k] + this.hitU * len) / this.arcTotal
    const u = s * L
    const shape = this.shape
    const w = this.halfWidthAt(s, L)
    this.hitS = s
    this.hitV = a * bestSign

    const head = this.shape.head
    if (head && u < L * (head.sweep + head.depth + head.notch)) {
      // Cephalofoil: a wide wing across the snout, its tips swept back toward the eyes,
      // and on a great hammerhead a shallow notch in the middle of the front edge.
      const span = L * head.span
      const r = a / span
      const front = L * head.sweep * r * r + L * head.notch * Math.max(0, 1 - r / 0.14)
      // Full depth almost to the tips, so the wing reads as a solid bar rather than a thin line.
      const depth = L * head.depth * (1 - 0.2 * r * r * r * r)
      if (r <= 1 && u >= front && u <= front + depth) {
        this.hitFlag = r > 0.84 ? FLAG_EYE : FLAG_BODY
        this.hitShade = r > 0.84 ? 0 : 0.62 - 0.2 * r + (u < front + L * 0.012 ? 0.1 : 0)
        return true
      }
    }

    if (a <= w) {
      const across = a / (w || 1)
      // Rounded back lit from the top-left; flanks fall off into shadow.
      const facing = (bestNx * -0.6 + bestNy * -0.8) * across
      let shade = 0.58 - across * across * 0.42 + facing * 0.26
      let flag = FLAG_BODY
      if (s > shape.dorsal[0] && s < shape.dorsal[1] && across < 0.3) {
        // First dorsal fin, catching the light.
        flag = FLAG_RIDGE
        shade = 1
      }
      else if (s > shape.secondDorsal[0] && s < shape.secondDorsal[1] && across < 0.4) {
        shade = Math.max(shade, 0.82)
      }
      else if (shape.eye && s > shape.eye[0] && s < shape.eye[1] && across > 0.6) {
        flag = FLAG_EYE
        shade = 0
      }
      else if (s > shape.gills[0] && s < shape.gills[1] && across > 0.58) {
        // Five gill slits.
        if (Math.floor(((s - shape.gills[0]) / (shape.gills[1] - shape.gills[0])) * 5) % 2 === 0) shade *= 0.45
      }
      else if (s < 0.05) {
        // Pale snout tip.
        shade = Math.max(shade, 0.7)
      }
      else if (shape === SHAPES.zebra && s > 0.08 && s < 0.55 && (across < 0.1 || Math.abs(across - 0.6) < 0.08)) {
        // Zebra sharks carry pale ridges down the back.
        shade = Math.max(shade, 0.8)
      }
      else if (this.marked(s, a * bestSign, L, across)) {
        flag = FLAG_MARK
      }
      this.hitFlag = flag
      this.hitShade = shade
      return true
    }

    const tip = bestSign > 0 ? this.finTipRight : this.finTipLeft
    const [p0, p1, p2] = shape.pectoral
    if (
      inTriangle(u, a, L * p0, 0, L * p1, 0, L * p2, tip)
      // Falcate (sickle) fins get a concave trailing edge.
      && !(shape.falcate && inTriangle(u, a, L * p1, 0, L * (p1 + (p2 - p1) * 0.33), tip * 0.55, L * p2, tip))
    ) {
      this.hitFlag = this.marked(s, a * bestSign, L, 0) ? FLAG_MARK : FLAG_FIN
      this.hitShade = 0.38 + 0.22 * (1 - clamp(a / tip, 0, 1))
      return true
    }
    const [v0, v1, v2, vr] = shape.pelvic
    if (
      inTriangle(u, a, L * v0, 0, L * v1, 0, L * v2, L * vr)
      // Forked caudal fin.
      || (shape.caudal > 0 && inTriangle(u, a, L * 0.86, 0, L * 0.95, 0, L, L * shape.caudal))
    ) {
      this.hitFlag = FLAG_FIN
      this.hitShade = 0.36
      return true
    }
    return false
  }

  /** Whether a body point (s along, v across) falls on a dark marking. */
  private marked(s: number, v: number, L: number, across: number) {
    if (this.shape === SHAPES.tiger) {
      // Wavy vertical bars over the back half, fading out toward the head.
      if (s < 0.2 || s > 0.9 || across > 0.85) return false
      const wobble = 0.35 * Math.sin((v / L) * 25 + s * 5)
      const bar = Math.sin(s * Math.PI * 2 * 9 + wobble)
      return bar > 0.65 - 0.5 * smoothstep(0.2, 0.42, s) + 0.4 * smoothstep(0.8, 0.9, s)
    }
    if (this.shape === SHAPES.zebra) {
      // Leopard spots on a jittered grid, so they ride along with the skin.
      if (s < 0.07) return false
      const gu = (s * L) / (L * 0.06)
      const gv = v / (L * 0.045)
      const cu = Math.floor(gu)
      const cv = Math.floor(gv)
      const jx = hash3(cu, cv, this.sharkId) * 0.5 + 0.25
      const jy = hash3(cu, cv, this.sharkId + 11) * 0.5 + 0.25
      return (gu - cu - jx) ** 2 + (gv - cv - jy) ** 2 < 0.08
    }
    return false
  }

  /**
   * Build a smooth, undulating render spine from the sim spine, then
   * supersample the cells within reach of each spine segment against the shape.
   * Fully covered cells become solid skin; partial cells become edge glyphs.
   */
  private rasterizeSharks(sharks: Shark[], t: number, frame: number) {
    const { sharkFlag, sharkShade, sharkKey, sharkDepth, sharkKind, sharkStamp, shadowStamp, sharkCells, visit } = this
    const { spine, arc, reach, reachRaw, cw, ch, cols, rows } = this
    const N = SHARK_NODES
    // From a cell's centre to beyond its farthest subsample.
    const half = Math.hypot(cw, ch) / 2
    let count = 0

    for (const sh of sharks) {
      const L = sh.length
      const n = sh.nodes
      const swim = clamp(sh.speed / 60, 0.4, 1.4)
      const tables = tablesFor(sh.kind)
      this.shape = SHAPES[sh.kind]
      this.widths = tables.width
      this.sharkId = sh.id
      const kindIndex = KINDS.indexOf(sh.kind)

      // Travelling body wave, growing toward the tail.
      for (let k = 0; k < N; k++) {
        const s = k / (N - 1)
        const a = Math.max(0, k - 1)
        const b = Math.min(N - 1, k + 1)
        let tx = n[a * 2] - n[b * 2]
        let ty = n[a * 2 + 1] - n[b * 2 + 1]
        const tl = Math.hypot(tx, ty) || 1
        tx /= tl
        ty /= tl
        const amp = L * 0.05 * (0.1 + 0.9 * Math.pow(s, 1.8)) * swim
        const off = amp * Math.sin(sh.phase - s * 5.2)
        spine[k * 2] = n[k * 2] - ty * off
        spine[k * 2 + 1] = n[k * 2 + 1] + tx * off
      }
      arc[0] = 0
      for (let k = 1; k < N; k++) {
        arc[k]
          = arc[k - 1]
            + Math.hypot(spine[k * 2] - spine[(k - 1) * 2], spine[k * 2 + 1] - spine[(k - 1) * 2 + 1])
      }
      this.arcTotal = arc[N - 1] || 1
      for (let k = 0; k < N - 1; k++) {
        const sx = spine[(k + 1) * 2] - spine[k * 2]
        const sy = spine[(k + 1) * 2 + 1] - spine[k * 2 + 1]
        const len2 = sx * sx + sy * sy || 1
        this.segAx[k] = spine[k * 2]
        this.segAy[k] = spine[k * 2 + 1]
        this.segSx[k] = sx
        this.segSy[k] = sy
        this.segLen[k] = Math.sqrt(len2)
        this.segInv[k] = 1 / len2
      }

      // Bank into turns: the inside fin foreshortens, the outside one spreads.
      const bank = clamp(sh.turn * 0.45, -0.35, 0.35)
      const flex = 1 + 0.06 * Math.sin(sh.phase * 0.5)
      const reachFin = this.shape.pectoral[3]
      this.finTipRight = L * reachFin * flex * (1 - bank)
      this.finTipLeft = L * reachFin * flex * (1 + bank)

      // How far the silhouette reaches from each 1/64 of the spine: the pectorals' leading edge runs from the root
      // out to the tip. A subsample's nearest spine point can sit a little along from its cell centre's, so the
      // reach checked after sampling a centre (reach) also covers the neighbouring bins.
      const tip = Math.max(this.finTipLeft, this.finTipRight)
      const [p0, , p2] = this.shape.pectoral
      for (let b = 0; b < REACH_BINS; b++) {
        const s1 = (b + 1) / REACH_BINS
        const pect = s1 > p0 && b / REACH_BINS < p2 ? tip * Math.min(1, (s1 - p0) / (p2 - p0)) : 0
        reachRaw[b] = Math.max(tables.reach[b] * L, pect)
      }
      const spread = Math.ceil(((3 * half) / L) * REACH_BINS)
      for (let b = 0; b < REACH_BINS; b++) {
        let r = 0
        for (let j = Math.max(0, b - spread); j <= Math.min(REACH_BINS - 1, b + spread); j++) r = Math.max(r, reachRaw[j])
        reach[b] = r + half
      }

      const shadowDc = Math.round((L * 0.035) / cw) + 1
      const shadowDr = Math.round((L * 0.02) / ch) + 1
      const visitId = ++this.visitId

      // A cell can only be on the shark if its centre is within reach of the segment nearest to what it covers,
      // so each segment visits the cells in its capsule, and every cell is sampled once.
      for (let k = 0; k < N - 1; k++) {
        const b0 = Math.min(REACH_BINS - 1, ((arc[k] / this.arcTotal) * REACH_BINS) | 0)
        const b1 = Math.min(REACH_BINS - 1, ((arc[k + 1] / this.arcTotal) * REACH_BINS) | 0)
        let r = 0
        for (let b = b0; b <= b1; b++) r = Math.max(r, reachRaw[b])
        r += half
        const r2 = r * r
        const ax = this.segAx[k]
        const ay = this.segAy[k]
        const sx = this.segSx[k]
        const sy = this.segSy[k]
        const inv = this.segInv[k]
        const c0 = Math.max(0, Math.floor((Math.min(ax, ax + sx) - r) / cw))
        const c1 = Math.min(cols - 1, Math.floor((Math.max(ax, ax + sx) + r) / cw))
        const r0 = Math.max(0, Math.floor((Math.min(ay, ay + sy) - r) / ch))
        const r1 = Math.min(rows - 1, Math.floor((Math.max(ay, ay + sy) + r) / ch))

        for (let row = r0; row <= r1; row++) {
          const y = row * ch
          const dy = y + ch / 2 - ay
          for (let c = c0; c <= c1; c++) {
            const i = row * cols + c
            if (visit[i] === visitId) continue
            const x = c * cw
            const dx = x + cw / 2 - ax
            const u = clamp((dx * sx + dy * sy) * inv, 0, 1)
            if ((dx - sx * u) ** 2 + (dy - sy * u) ** 2 > r2) continue
            visit[i] = visitId

            const centerHit = this.sampleShark(x + cw / 2, y + ch / 2, L)
            if (!centerHit && this.hitDist > reach[Math.min(REACH_BINS - 1, (this.hitS * REACH_BINS) | 0)]) continue
            const flag = this.hitFlag
            const shade = this.hitShade
            const s = this.hitS
            const v = this.hitV
            // Subsamples sit within a cell of the centre: their nearest spine point is on a neighbouring segment.
            const k0 = Math.max(0, this.hitK - 2)
            const k1 = Math.min(N - 1, this.hitK + 3)

            let hits = centerHit ? 1 : 0
            for (let q = 0; q < SUBSAMPLES.length; q += 2) {
              if (this.inside(x + SUBSAMPLES[q] * cw, y + SUBSAMPLES[q + 1] * ch, L, k0, k1)) hits++
            }
            if (hits === 0) continue
            const coverage = hits / 5

            if (sharkStamp[i] !== frame) {
              sharkStamp[i] = frame
              sharkCells[count++] = i
            }
            // Body-anchored key so each glyph rides along with the skin.
            const key = (hash3(Math.round(s * 52), Math.round(v / (cw * 0.9)), sh.id) * 4294967295) >>> 0
            sharkKey[i] = key
            sharkDepth[i] = sh.depth
            sharkKind[i] = kindIndex

            if (centerHit && coverage >= 0.6) {
              sharkFlag[i] = flag
              const flicker = (hash3(c, row, Math.floor(t * 6)) - 0.5) * 0.06
              sharkShade[i] = clamp(shade + flicker, 0, 1)
              const sc = c + shadowDc
              const sr = row + shadowDr
              if (sc < cols && sr < rows) shadowStamp[sr * cols + sc] = frame
            }
            else {
              sharkFlag[i] = FLAG_EDGE
              sharkShade[i] = coverage
            }
          }
        }
      }
    }
    this.sharkCellCount = count
  }

  /** Draws a sprite over cell i, in device px. */
  private stampSprite(i: number, sprite: number) {
    const { atlas, cols } = this
    this.ctx.drawImage(
      atlas.canvas, atlas.x[sprite], atlas.y[sprite], atlas.w, atlas.h,
      this.colX[i % cols] - atlas.pad, this.rowY[(i / cols) | 0] - atlas.pad, atlas.w, atlas.h,
    )
  }

  private drawSharks(t: number) {
    const { ctx, atlas, cols, cw, ch, dpr, sharkCells, sharkCellCount: count, sharkFlag, sharkShade, sharkKey } = this
    const { sharkDepth, sharkKind, cellBlock, cellSprite, blockCounts, blockCursor, blockOrder, blocksUsed } = this
    if (count === 0) return
    const accents = this.accents

    for (let k = 0; k < count; k++) {
      const i = sharkCells[k]
      const flag = sharkFlag[i]
      const key = sharkKey[i]
      const shade = sharkShade[i]
      const dim = 0.7 + 0.3 * sharkDepth[i]
      const kind = sharkKind[i]

      // Colours re-roll on a staggered clock so the hide shimmers.
      const tick = Math.floor(t * 2.4 + (key % 997) / 97)
      const roll = hash3(key, tick, 7)

      if (flag === FLAG_EDGE) {
        // Soft, anti-aliased silhouette: no block, just a light glyph.
        const set = shade >= 0.5 ? EDGE_HEAVY : EDGE_LIGHT
        const g = (70 + shade * 80) * dim
        const color = roll < 0.07 ? accents[key % accents.length] : this.edgeColors[kind][Math.min(EDGE_LEVELS - 1, Math.round(g / 2))]
        cellBlock[k] = -1
        cellSprite[k] = atlas.sprite(set[(key >>> 8) % set.length], color)
        continue
      }

      let level: number
      let glyph: number
      if (flag === FLAG_RIDGE) {
        level = 226
        glyph = RIDGE_GLYPHS[(key >>> 8) % RIDGE_GLYPHS.length]
      }
      else if (flag === FLAG_EYE) {
        level = 12
        glyph = EYE_GLYPH
      }
      else if (flag === FLAG_MARK) {
        level = 24 + shade * 22
        glyph = MARK_GLYPHS[(key >>> 8) % MARK_GLYPHS.length]
      }
      else if (flag === FLAG_FIN) {
        level = 40 + shade * 120
        glyph = FIN_GLYPHS[(key >>> 8) % FIN_GLYPHS.length]
      }
      else {
        level = 46 + shade * 160
        const jitter = ((key >>> 4) % 3) - 1
        glyph = SKIN_RAMP[clamp(Math.floor((1 - shade) * 8.99) + jitter, 0, SKIN_RAMP.length - 1)]
      }
      level *= dim
      // Tinted per species; the classic tint is a cool blue-grey, like a back seen through deep water.
      const block = kind * LEVELS + Math.min(LEVELS - 1, Math.round(level / 2))
      cellBlock[k] = block
      if (blockCounts[block]++ === 0) blocksUsed.push(block)

      let color: number
      if (flag === FLAG_EYE) color = this.eye
      else if (flag === FLAG_MARK) color = this.markColors[kind]
      else if (roll < 0.09) color = accents[key % accents.length]
      else if (roll < 0.13) color = this.glint
      else color = shade > 0.55 ? this.inkLit : this.inkShade
      cellSprite[k] = atlas.sprite(glyph, color)
    }

    // The blocks, one fill per shade: counting sort the cells by fill.
    let start = 0
    for (const b of blocksUsed) {
      blockCursor[b] = start
      start += blockCounts[b]
    }
    for (let k = 0; k < count; k++) {
      const b = cellBlock[k]
      if (b >= 0) blockOrder[blockCursor[b]++] = sharkCells[k]
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    let at = 0
    for (const b of blocksUsed) {
      ctx.fillStyle = this.blockStyles[b]
      for (const end = at + blockCounts[b]; at < end; at++) {
        const i = blockOrder[at]
        ctx.fillRect((i % cols) * cw + 1, ((i / cols) | 0) * ch + 1, cw - 1, ch - 1)
      }
      blockCounts[b] = 0
    }
    blocksUsed.length = 0

    // Then every glyph on top.
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    for (let k = 0; k < count; k++) this.stampSprite(sharkCells[k], cellSprite[k])
  }

  // ── fish, dots, heat ────────────────────────────────────────────────

  /**
   * Snap each fish to a cell and bucket it by brightness: far fish are dim,
   * near fish brighter, and turning or panicking fish glint. Nearer fish
   * claim a cell first when two land in the same one.
   */
  private markFish(world: World, frame: number) {
    const { fishStamp, fishBuckets, sharkStamp } = this
    for (const bucket of fishBuckets) bucket.length = 0
    for (let pass = 0; pass < 2; pass++) {
      for (const f of world.fish) {
        if (!f.alive || (f.depth > 0.78 ? 0 : 1) !== pass) continue
        const i = this.cellIndex(f.x, f.y)
        if (i < 0 || fishStamp[i] === frame || sharkStamp[i] === frame) continue
        fishStamp[i] = frame
        const glint = clamp(Math.abs(f.turn) / 3.5, 0, 1)
        const v = 0.3 + (0.4 * (f.depth - 0.55)) / 0.45 + 0.3 * Math.max(glint, f.alarm)
        fishBuckets[clamp(Math.floor(v * FISH_ALPHAS.length), 0, FISH_ALPHAS.length - 1)].push(i)
      }
    }
  }

  /** Where the fish glyph's ink sits in a cell, measured from the font: textBaseline middle, centred. */
  private measureDash() {
    const { ctx, cw, ch, dpr } = this
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.font = `600 ${Math.round(ch * 0.74) * dpr}px ${this.fontFamily}`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    const m = ctx.measureText(String.fromCharCode(FISH_GLYPH))
    const ox = (cw * dpr) / 2
    const oy = (ch / 2 + 1) * dpr
    this.dash = {
      x: ox - m.actualBoundingBoxLeft,
      y: oy - m.actualBoundingBoxAscent,
      w: m.actualBoundingBoxLeft + m.actualBoundingBoxRight,
      h: m.actualBoundingBoxAscent + m.actualBoundingBoxDescent,
    }
  }

  private drawFish() {
    const { ctx, fishBuckets, colX, rowY, cols, dash } = this
    for (let b = 0; b < fishBuckets.length; b++) {
      const bucket = fishBuckets[b]
      if (!bucket.length) continue
      ctx.fillStyle = this.fishStyles[b]
      for (const i of bucket) ctx.fillRect(colX[i % cols] + dash.x, rowY[(i / cols) | 0] + dash.y, dash.w, dash.h)
    }
  }

  private drawHeat() {
    const { heat } = this
    for (let i = 0; i < heat.length; i++) {
      const h = heat[i]
      if (h < 0.08) continue
      const hot = h > 0.5
      const alpha = Math.min(HEAT_LEVELS, Math.round((hot ? h : h * 1.6) * HEAT_LEVELS))
      const color = hot ? this.heatHot[alpha] : this.heatWarm[alpha]
      this.stampSprite(i, this.atlas.sprite(h > 0.6 ? HEAT_HOT_GLYPH : HEAT_GLYPH, color))
    }
  }

  /**
   * Every cell's dot twinkles on its own clock, is modulated by slow
   * caustic light bands, and flares where the wake field is strong.
   * Dots are bucketed by colour/alpha so each bucket is one fillStyle.
   */
  private drawDots(t: number, frame: number) {
    const { ctx, cols, rows, cw, ch, dotKind, dotPhase, dotFreq, energy, sharkStamp, shadowStamp, fishStamp } = this
    const { bucketOf, bucketOrder, bucketCounts, bucketStarts: starts, bucketCursor: cursor } = this
    const n = cols * rows
    const SKIP = bucketCounts.length - 1
    bucketCounts.fill(0)

    for (let i = 0; i < n; i++) {
      const e = energy[i]
      const kind = dotKind[i]
      if (sharkStamp[i] === frame || fishStamp[i] === frame || (kind === 0 && e < 0.45)) {
        bucketOf[i] = SKIP
        continue
      }
      let bucket: number
      if (e > 0.45) {
        const level = clamp(Math.floor((e - 0.45) * 5), 0, ALPHA_LEVELS - 1)
        bucket = WAKE * ALPHA_LEVELS + level
      }
      else {
        const col = i % cols
        const row = (i / cols) | 0
        // Dots stay bright and crisp; the twinkle is a gentle breathe, not a fade.
        const twinkle = 0.5 + 0.5 * Math.sin(t * dotFreq[i] + dotPhase[i])
        const caustic = 0.85 + 0.15 * Math.sin(col * 0.11 + t * 0.35) * Math.sin(row * 0.17 - t * 0.27)
        let v = clamp((0.55 + 0.45 * twinkle) * caustic + e * 2, 0, 1)
        if (shadowStamp[i] === frame) v *= 0.2
        bucket = (kind - 1) * ALPHA_LEVELS + clamp(Math.floor(v * ALPHA_LEVELS), 0, ALPHA_LEVELS - 1)
      }
      bucketOf[i] = bucket
      bucketCounts[bucket]++
    }

    // Counting sort cells by bucket.
    starts[0] = 0
    for (let b = 1; b < bucketCounts.length; b++) starts[b] = starts[b - 1] + bucketCounts[b - 1]
    cursor.set(starts)
    for (let i = 0; i < n; i++) {
      const b = bucketOf[i]
      if (b !== SKIP) bucketOrder[cursor[b]++] = i
    }

    const small = Math.max(2, Math.round(cw * 0.19))
    const large = small + 1
    for (let b = 0; b < SKIP; b++) {
      const count = bucketCounts[b]
      if (!count) continue
      ctx.fillStyle = this.dotStyles[b]
      const size = b >= WAKE * ALPHA_LEVELS ? large : small
      const start = starts[b]
      for (let k = start; k < start + count; k++) {
        const i = bucketOrder[k]
        const col = i % cols
        const row = (i / cols) | 0
        // Sit the dot low-right in its cell, as in the reference.
        ctx.fillRect(Math.round(col * cw + cw * 0.66 - size / 2), Math.round(row * ch + ch * 0.64 - size / 2), size, size)
      }
    }
  }

  /**
   * Lays the bare water back over the quiet rects, so what swims under the copy shows at palette.quiet at most
   * while the ground and grid stay as they are everywhere else.
   */
  private drawVeil() {
    const { ctx, veil, veilEnds, bg } = this
    if (!veil.length) return
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    let k = 0
    for (let level = 0; level < 3; level++) {
      ctx.globalAlpha = 1 - (level === 0 ? this.palette.quiet : RING_ALPHA[level - 1])
      for (; k < veilEnds[level]; k += 4) {
        const x = veil[k]
        const y = veil[k + 1]
        ctx.drawImage(bg, x, y, veil[k + 2], veil[k + 3], x, y, veil[k + 2], veil[k + 3])
      }
    }
    ctx.globalAlpha = 1
  }
}
