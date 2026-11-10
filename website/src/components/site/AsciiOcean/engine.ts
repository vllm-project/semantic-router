// Adapted from stackedlol/ascii-sharks (MIT); see LICENSE in this folder.
import { clamp } from './math'
import { PALETTES, type PaletteName } from './palettes'
import { AsciiRenderer } from './renderer'
import { World, type SharkSpec } from './world'

const STEP = 1 / 60
const MAX_STEPS = 4
/** A frame this close to a whole step takes it, so a 60 Hz display steps once a frame despite timer jitter. */
const SNAP = 0.002
/** How often the quiet rects are measured again (ms): web fonts and entrance animations move the copy. */
const QUIET_EVERY = 500
/** Reduced motion settles the sim in slices of this many ms, so it never blocks the page. */
const SETTLE_SLICE = 8
/**
 * The first frames take one step at most: while the code is still cold, a frame that caught up on the last one's
 * steps would run long in turn.
 */
const WARM_FRAMES = 30

/** A rect in viewport coordinates; DOMRect is one. */
export type QuietRect = Pick<DOMRectReadOnly, 'left' | 'top' | 'right' | 'bottom'>

export interface OceanOptions {
  /** The sharks in the water; size is relative to the biggest. */
  sharks?: SharkSpec[]
  /** Multiplies the area-based fish population. */
  density?: number
  /** 'auto' follows the site theme (html[data-theme]): paper in light, night in dark. */
  palette?: PaletteName | 'auto'
  /** The hero copy's rects: cells under them draw faint, and fish and sharks keep clear. */
  quietRects?: () => readonly QuietRect[]
}

const INTERACTIVE = 'a, button, input, select, textarea, label, summary, [role="button"]'

/**
 * Wires World + AsciiRenderer to a canvas: fixed-timestep loop, resize,
 * pointer input, the quiet rects, pausing off-screen and in hidden tabs,
 * and prefers-reduced-motion.
 */
export class OceanEngine {
  private world: World
  private renderer: AsciiRenderer
  private raf = 0
  private last = 0
  private acc = 0
  private visible = true
  private reducedMotion: MediaQueryList
  private resizeObserver: ResizeObserver
  private intersectionObserver: IntersectionObserver
  private themeObserver: MutationObserver | null = null
  private disposed = false
  private palette: PaletteName | 'auto'
  private quietRects?: () => readonly QuietRect[]
  private quiet = new Float32Array(0)
  private quietAt = -Infinity
  private settleTimer = 0
  private startTimer = 0
  private size = ''
  private started = false
  private frames = 0

  constructor(private canvas: HTMLCanvasElement, options: OceanOptions = {}) {
    this.world = new World(options.sharks, undefined, options.density)
    this.palette = options.palette ?? 'auto'
    this.quietRects = options.quietRects
    this.renderer = new AsciiRenderer(canvas, PALETTES[this.resolvePalette()])
    this.renderer.setFontFamily(getComputedStyle(canvas).fontFamily)
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')

    this.resizeObserver = new ResizeObserver(() => this.resize())
    this.resizeObserver.observe(canvas)
    this.intersectionObserver = new IntersectionObserver((entries) => {
      this.visible = entries[entries.length - 1].isIntersecting
      this.schedule()
    })
    this.intersectionObserver.observe(canvas)
    this.observeTheme()

    this.reducedMotion.addEventListener('change', this.onMotionPreference)
    document.addEventListener('visibilitychange', this.onVisibility)
    window.addEventListener('pointermove', this.onPointerMove, { passive: true })
    window.addEventListener('pointerdown', this.onPointerDown, { passive: true })
    document.documentElement.addEventListener('pointerleave', this.onPointerLeave)
    window.addEventListener('blur', this.onPointerLeave)

    this.resize()
    // Canvas text can't trigger a webfont load. AsciiOcean starts the load before the engine, so the face is
    // usually in already; if not, redraw once it is.
    const font = `600 16px ${getComputedStyle(canvas).fontFamily}`
    if (document.fonts && !document.fonts.check(font)) {
      document.fonts.load(font).then(() => {
        if (this.disposed) return
        this.renderer.refreshGlyphs()
        if (this.reducedMotion.matches) this.drawStill()
      }, () => undefined)
    }
    // The page's own fonts reflow the copy.
    document.fonts?.ready.then(() => {
      if (!this.disposed) this.measureQuiet()
    }, () => undefined)
    this.startTimer = window.setTimeout(this.start, 0)
  }

  destroy() {
    this.disposed = true
    cancelAnimationFrame(this.raf)
    window.clearTimeout(this.settleTimer)
    window.clearTimeout(this.startTimer)
    this.resizeObserver.disconnect()
    this.intersectionObserver.disconnect()
    this.themeObserver?.disconnect()
    this.reducedMotion.removeEventListener('change', this.onMotionPreference)
    document.removeEventListener('visibilitychange', this.onVisibility)
    window.removeEventListener('pointermove', this.onPointerMove)
    window.removeEventListener('pointerdown', this.onPointerDown)
    document.documentElement.removeEventListener('pointerleave', this.onPointerLeave)
    window.removeEventListener('blur', this.onPointerLeave)
    this.renderer.destroy()
  }

  setPalette(palette: PaletteName | 'auto') {
    this.palette = palette
    this.observeTheme()
    this.applyPalette()
  }

  private resolvePalette(): PaletteName {
    if (this.palette !== 'auto') return this.palette
    return document.documentElement.dataset.theme === 'dark' ? 'night' : 'paper'
  }

  private observeTheme() {
    this.themeObserver?.disconnect()
    this.themeObserver = null
    if (this.palette !== 'auto') return
    this.themeObserver = new MutationObserver(() => this.applyPalette())
    this.themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
  }

  private applyPalette() {
    this.renderer.setPalette(PALETTES[this.resolvePalette()])
    this.repaint()
  }

  private resize() {
    const rect = this.canvas.getBoundingClientRect()
    if (rect.width < 1 || rect.height < 1) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    // The observer's first callback, and any that leave the box as it was, would redo all of it for nothing.
    const size = `${rect.width}x${rect.height}@${dpr}`
    if (size === this.size) return
    this.size = size
    this.renderer.resize(rect.width, rect.height, dpr)
    // Measured before the first resize of the world, which seeds the schools clear of the copy.
    this.measureQuiet(true)
    this.world.resize(rect.width, rect.height, this.renderer.cellWidth, this.renderer.cellHeight)
    this.repaint()
  }

  /**
   * Draws the current state now. A resized canvas is opaque black until its next frame, and a paused one would
   * otherwise come back on screen showing its old palette for a frame.
   */
  private repaint() {
    if (this.reducedMotion.matches) this.drawStill()
    // Until the first frame, the bare water: setting up and the first render then fall in separate tasks.
    else if (this.started) this.renderer.render(this.world)
    else this.renderer.clear()
  }

  /** The quiet rects in canvas px, handed to the world and the renderer when they change. */
  private measureQuiet(force = false) {
    this.quietAt = performance.now()
    if (!this.quietRects) return
    const box = this.canvas.getBoundingClientRect()
    const local: number[] = []
    for (const r of this.quietRects()) {
      const x0 = Math.max(0, r.left - box.left)
      const y0 = Math.max(0, r.top - box.top)
      const x1 = Math.min(box.width, r.right - box.left)
      const y1 = Math.min(box.height, r.bottom - box.top)
      if (x1 - x0 >= 1 && y1 - y0 >= 1) local.push(x0, y0, x1, y1)
    }
    const same = local.length === this.quiet.length && local.every((v, i) => Math.abs(v - this.quiet[i]) < 0.5)
    if (same && !force) return
    this.quiet = new Float32Array(local)
    this.world.setQuiet(this.quiet)
    this.renderer.setQuiet(this.quiet)
  }

  /**
   * While the code is cold, the first sim step and then the first render, each in a task of its own rather than
   * beside the browser's rendering work; then the loop.
   */
  private start = () => {
    if (this.disposed) return
    if (!this.started && !this.reducedMotion.matches) {
      if (this.world.time === 0) {
        this.world.step(STEP)
        this.renderer.advance(this.world, STEP)
        this.startTimer = window.setTimeout(this.start, 0)
        return
      }
      this.renderer.render(this.world)
      this.started = true
    }
    this.schedule()
  }

  private schedule() {
    cancelAnimationFrame(this.raf)
    if (this.disposed || !this.started || !this.visible || document.hidden || this.reducedMotion.matches) return
    this.last = performance.now()
    this.raf = requestAnimationFrame(this.frame)
  }

  private frame = (now: number) => {
    const dt = clamp((now - this.last) / 1000, 0, 0.1)
    this.last = now
    this.acc += dt
    const maxSteps = this.frames++ < WARM_FRAMES ? 1 : MAX_STEPS
    let steps = 0
    while (this.acc >= STEP - SNAP && steps < maxSteps) {
      this.world.step(STEP)
      this.renderer.advance(this.world, STEP)
      this.acc -= STEP
      steps++
    }
    if (steps === maxSteps) this.acc = 0
    if (now - this.quietAt > QUIET_EVERY) this.measureQuiet()
    // The sim moves in whole steps; a frame without one would draw the same picture again.
    if (steps > 0) this.renderer.render(this.world)
    this.raf = requestAnimationFrame(this.frame)
  }

  /** Reduced motion: let the ecosystem settle, then show a single frame. */
  private drawStill() {
    window.clearTimeout(this.settleTimer)
    // Every slice, and the still frame after them, is a task of its own: none joins its caller's.
    const settle = () => {
      if (this.disposed) return
      if (this.world.time >= 4) {
        this.renderer.render(this.world)
        return
      }
      const until = performance.now() + SETTLE_SLICE
      while (this.world.time < 4 && performance.now() < until) {
        this.world.step(STEP)
        this.renderer.advance(this.world, STEP)
      }
      this.settleTimer = window.setTimeout(settle, 0)
    }
    if (this.world.time >= 4) {
      settle()
      return
    }
    // Bare water while it settles, rather than the unsettled scene and then a jump.
    this.renderer.clear()
    this.settleTimer = window.setTimeout(settle, 0)
  }

  private onMotionPreference = () => {
    if (this.reducedMotion.matches) {
      cancelAnimationFrame(this.raf)
      this.drawStill()
    }
    else {
      window.clearTimeout(this.settleTimer)
      this.start()
    }
  }

  private onVisibility = () => {
    if (document.hidden) cancelAnimationFrame(this.raf)
    else this.schedule()
  }

  private toLocal(e: PointerEvent) {
    const rect = this.canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const inside = x >= 0 && y >= 0 && x <= rect.width && y <= rect.height
    return { x, y, inside }
  }

  private onPointerMove = (e: PointerEvent) => {
    const { x, y, inside } = this.toLocal(e)
    const p = this.world.pointer
    p.x = x
    p.y = y
    p.active = inside && e.pointerType !== 'touch'
  }

  private onPointerDown = (e: PointerEvent) => {
    const { x, y, inside } = this.toLocal(e)
    // A click on the hero's links and buttons is theirs, not a splash.
    const onControl = e.target instanceof Element && e.target.closest(INTERACTIVE)
    if (inside && !onControl && !this.reducedMotion.matches) this.world.splash(x, y)
  }

  private onPointerLeave = () => {
    this.world.pointer.active = false
  }
}
