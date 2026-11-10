import { numberVar, onThemeChange, reducedMotion, rgbVar, rng } from './pixels'

/* One canvas engine for the homepage's interludes, in Duo's 8 px blocks on a half-resolution backing store:
 *   dawn  — blocks rise off a horizon into the cells of the serif line, then dissolve as the real text resolves;
 *   night — dithered ink edges and pixel stars that twinkle a few at a time;
 *   ember — a left-to-right mosaic wipe of ember tiles behind a quote.
 * It plays only while the band is on screen and the tab is visible, and draws the finished frame at once under
 * reduced motion. The host's data-state ("wait", then "show") tells the CSS when its text may appear. */

export type InterludeKind = 'dawn' | 'night' | 'ember'

const K = 2
const U = 4
const CELL = U * K

type Frame = { more: boolean, show: boolean }

interface Scene {
  // Lays the scene out for a size, in cells.
  build: (cols: number, rows: number, w: number, h: number) => void
  // Re-reads the colours.
  paint: () => void
  // Draws the frame at t ms of play (Infinity for the finished frame).
  draw: (t: number) => Frame
  // Frames per second while it plays; 0 means every animation frame.
  fps: number
  // Whether the text is shown before the band first plays.
  showFirst: boolean
}

function easeOut(x: number): number {
  const t = Math.max(0, Math.min(1, x))
  return 1 - (1 - t) ** 3
}

/* Words of the line, rasterised at their rendered positions, sampled into cells. */
function lineCells(host: HTMLElement, cols: number, rows: number, w: number, h: number): [number, number][] {
  const line = host.querySelector<HTMLElement>('[data-line]')
  if (!line) return []
  const off = document.createElement('canvas')
  off.width = w
  off.height = h
  const o = off.getContext('2d', { willReadFrequently: true })
  if (!o) return []
  const cs = getComputedStyle(line)
  o.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`
  if ('letterSpacing' in o) (o as CanvasRenderingContext2D & { letterSpacing: string }).letterSpacing = cs.letterSpacing
  o.textBaseline = 'alphabetic'
  o.fillStyle = '#000'
  const origin = host.getBoundingClientRect()
  line.querySelectorAll<HTMLElement>('[data-word]').forEach((word) => {
    const text = word.textContent ?? ''
    const r = word.getBoundingClientRect()
    const m = o.measureText(text)
    const ascent = m.fontBoundingBoxAscent || m.actualBoundingBoxAscent
    const descent = m.fontBoundingBoxDescent || m.actualBoundingBoxDescent
    const top = r.top - origin.top + (r.height - ascent - descent) / 2
    o.fillText(text, r.left - origin.left, top + ascent)
  })
  const data = o.getImageData(0, 0, w, h).data
  const cells: [number, number][] = []
  for (let cy = 0; cy < rows; cy++) {
    for (let cx = 0; cx < cols; cx++) {
      let sum = 0
      let n = 0
      for (let y = 1; y < CELL; y += 2) {
        for (let x = 1; x < CELL; x += 2) {
          const px = cx * CELL + x
          const py = cy * CELL + y
          if (px >= w || py >= h) continue
          sum += data[(py * w + px) * 4 + 3]
          n++
        }
      }
      if (n && sum / n / 255 > 0.3) cells.push([cx, cy])
    }
  }
  return cells
}

function dawnScene(ctx: CanvasRenderingContext2D, host: HTMLElement, seed: number): Scene {
  type Block = { x: number, y: number, from: number, delay: number, tone: number, fade: number }
  let blocks: Block[] = []
  let horizon: Block[] = []
  let tones: string[] = []
  let faint: string[] = []
  let cv = { cols: 0, rows: 0 }
  const RISE = 760
  const ARRIVE = 1350
  const DISSOLVE = 560

  return {
    fps: 0,
    showFirst: false,
    paint() {
      const boost = numberVar('--band-boost', 1)
      const hues = [1, 2, 3, 4].map(i => rgbVar(`--px-duo-${i}`, '208 62 26'))
      tones = hues.map(rgb => `rgba(${rgb}, ${Math.min(1, 0.78 * boost).toFixed(3)})`)
      faint = hues.map(rgb => `rgba(${rgb}, ${Math.min(1, 0.16 * boost).toFixed(3)})`)
    },
    build(cols, rows, w, h) {
      cv = { cols, rows }
      const r = rng(seed * 131 + cols)
      blocks = lineCells(host, cols, rows, w, h).map(([x, y]) => ({
        x,
        y,
        from: rows + 1 + Math.floor(r() * 6),
        delay: (x / cols) * 420 + r() * 160,
        tone: Math.max(0, Math.min(3, Math.floor((x / cols) * 4 + (r() - 0.5) * 0.6))),
        fade: r(),
      }))
      horizon = []
      for (let x = 0; x < cols; x++) {
        const edge = Math.abs(x / cols - 0.5) * 2
        const height = Math.round(2 + 4 * edge ** 1.4 + r() * 2)
        for (let k = 0; k < height; k++) {
          if (k === height - 1 && r() < 0.5) continue
          horizon.push({ x, y: rows - 1 - k, from: rows + 1, delay: r() * 380, tone: Math.min(3, Math.floor(k / 2)), fade: 0 })
        }
      }
    },
    draw(t) {
      ctx.clearRect(0, 0, cv.cols * U, cv.rows * U)
      const buckets: number[][][] = [[], [], [], [], [], [], [], []]
      for (const b of horizon) {
        const p = easeOut((t - b.delay) / 600)
        if (p <= 0) continue
        const y = Math.round(b.from + (b.y - b.from) * p)
        buckets[4 + b.tone].push([b.x, y])
      }
      for (const b of blocks) {
        if (t > ARRIVE && (t - ARRIVE) / DISSOLVE > b.fade) continue
        const p = easeOut((t - b.delay) / RISE)
        if (p <= 0) continue
        const y = Math.round(b.from + (b.y - b.from) * p)
        buckets[b.tone].push([b.x, y])
      }
      buckets.forEach((bucket, k) => {
        if (!bucket.length) return
        ctx.fillStyle = k < 4 ? tones[k] : faint[k - 4]
        for (const [x, y] of bucket) ctx.fillRect(x * U, y * U, U - 1, U - 1)
      })
      return { more: t < ARRIVE + DISSOLVE, show: t >= ARRIVE - 120 }
    },
  }
}

function nightScene(ctx: CanvasRenderingContext2D, seed: number): Scene {
  type Star = { x: number, y: number, s: number, color: string, base: number, level: number }
  let stars: Star[] = []
  let edges: [number, number][] = []
  let ink = ''
  let cv = { cols: 0, rows: 0 }
  let drawn = false
  let r = rng(seed)
  const LEVELS = [0.35, 0.7, 1]
  const EDGE = 4

  const starStyle = (star: Star) => `rgba(${star.color}, ${(star.base * LEVELS[star.level]).toFixed(3)})`
  const drawStar = (star: Star) => {
    ctx.clearRect(star.x, star.y, star.s, star.s)
    ctx.fillStyle = starStyle(star)
    ctx.fillRect(star.x, star.y, star.s, star.s)
  }

  return {
    fps: 6,
    showFirst: true,
    paint() {
      ink = `rgb(${rgbVar('--night-rgb', '22 24 29')})`
      drawn = false
    },
    build(cols, rows) {
      cv = { cols, rows }
      r = rng(seed * 977 + cols)
      edges = []
      for (let k = 0; k < EDGE; k++) {
        const p = (k + 1) / (EDGE + 1)
        for (let x = 0; x < cols; x++) {
          if (r() < p) edges.push([x, k])
          if (r() < p) edges.push([x, rows - 1 - k])
        }
      }
      stars = []
      const n = Math.round(cols * (rows - 2 * EDGE) / 26)
      for (let i = 0; i < n; i++) {
        const kind = r()
        stars.push({
          x: Math.floor(r() * cols * U),
          y: Math.floor((EDGE + 1 + r() * (rows - 2 * EDGE - 2)) * U),
          s: kind < 0.1 ? 2 : 1,
          color: kind < 0.04 ? rgbVar('--px-duo-1', '239 86 48') : kind < 0.08 ? '147, 179, 228' : '243, 239, 230',
          base: 0.35 + 0.6 * r(),
          level: Math.floor(r() * LEVELS.length),
        })
      }
      drawn = false
    },
    draw(t) {
      if (!drawn || !Number.isFinite(t)) {
        ctx.clearRect(0, 0, cv.cols * U, cv.rows * U)
        ctx.fillStyle = ink
        for (const [x, y] of edges) ctx.fillRect(x * U, y * U, U, U)
        for (const star of stars) {
          ctx.fillStyle = starStyle(star)
          ctx.fillRect(star.x, star.y, star.s, star.s)
        }
        drawn = true
        return { more: Number.isFinite(t), show: true }
      }
      const changes = Math.max(1, Math.round(stars.length * 0.04))
      for (let i = 0; i < changes; i++) {
        const star = stars[Math.floor(r() * stars.length)]
        star.level = (star.level + 1 + Math.floor(r() * 2)) % LEVELS.length
        drawStar(star)
      }
      return { more: true, show: true }
    },
  }
}

function emberScene(ctx: CanvasRenderingContext2D, seed: number): Scene {
  type Tile = { x: number, y: number, size: number, shade: number, at: number }
  let tiles: Tile[] = []
  let shades: string[] = []
  let cv = { cols: 0, rows: 0 }
  const WIPE = 1150
  const EDGE = 2

  return {
    fps: 0,
    showFirst: false,
    paint() {
      const [r0, g0, b0] = rgbVar('--interlude-ember', '201 63 29').split(', ').map(Number)
      shades = [1, 0.93, 0.87, 1.06].map(f => `rgb(${Math.min(255, Math.round(r0 * f))}, ${Math.round(g0 * f)}, ${Math.round(b0 * f)})`)
    },
    build(cols, rows) {
      cv = { cols, rows }
      const r = rng(seed * 613 + cols)
      tiles = []
      for (let y = 0; y < rows; y++) {
        const edge = y < EDGE || y >= rows - EDGE
        if (!edge && (y - EDGE) % 4) continue
        for (let x = 0; x < cols; x += edge ? 1 : 4) {
          if (edge) {
            const depth = y < EDGE ? y + 1 : rows - y
            if (r() > depth / (EDGE + 1)) continue
          }
          const at = (x / cols) * 0.82 + (y / rows) * 0.1 + r() * 0.08
          tiles.push({ x, y, size: edge ? 1 : Math.min(4, rows - EDGE - y), shade: Math.floor(r() * shades.length), at })
        }
      }
    },
    draw(t) {
      const p = t / WIPE
      ctx.clearRect(0, 0, cv.cols * U, cv.rows * U)
      const buckets: number[][][] = shades.map(() => [])
      for (const tile of tiles) {
        if (p < tile.at - 0.05) continue
        if (p < tile.at && tile.size > 1) {
          for (let k = 0; k < tile.size * tile.size; k += 3) {
            buckets[tile.shade].push([tile.x + (k % tile.size), tile.y + Math.floor(k / tile.size), 1])
          }
          continue
        }
        buckets[tile.shade].push([tile.x, tile.y, tile.size])
      }
      buckets.forEach((bucket, k) => {
        if (!bucket.length) return
        ctx.fillStyle = shades[k]
        for (const [x, y, size] of bucket) ctx.fillRect(x * U, y * U, size * U, size * U)
      })
      return { more: p < 1.05, show: p >= 0.7 }
    },
  }
}

export function mountInterlude(host: HTMLElement, kind: InterludeKind, seed: number): () => void {
  const cv = host.querySelector('canvas')
  const ctx = cv?.getContext('2d')
  if (!cv || !ctx) return () => undefined
  const scene = kind === 'dawn' ? dawnScene(ctx, host, seed) : kind === 'night' ? nightScene(ctx, seed) : emberScene(ctx, seed)
  const still = reducedMotion().matches
  let w = 0
  let h = 0
  let played = 0
  let last = 0
  let raf = 0
  let timer = 0
  let visible = false
  let started = false
  let finished = false
  let alive = true

  const setShow = (show: boolean) => {
    host.dataset.state = show ? 'show' : 'wait'
  }

  const size = (): boolean => {
    w = host.clientWidth
    h = host.clientHeight
    if (!w || !h) return false
    const cols = Math.ceil(w / CELL)
    const rows = Math.ceil(h / CELL)
    cv.width = cols * U
    cv.height = rows * U
    cv.style.width = `${cols * CELL}px`
    cv.style.height = `${rows * CELL}px`
    scene.build(cols, rows, w, h)
    return true
  }

  const stop = () => {
    if (raf) cancelAnimationFrame(raf)
    if (timer) window.clearTimeout(timer)
    raf = 0
    timer = 0
    last = 0
  }

  const tick = (now: number) => {
    raf = 0
    if (!alive || !visible || document.hidden) {
      last = 0
      return
    }
    if (last) played += Math.min(100, now - last)
    last = now
    const frame = scene.draw(played)
    if (frame.show) setShow(true)
    if (frame.more) schedule()
    else finished = true
  }

  function schedule() {
    if (raf || timer || finished) return
    if (scene.fps) {
      timer = window.setTimeout(() => {
        timer = 0
        raf = requestAnimationFrame(tick)
      }, 1000 / scene.fps)
    }
    else {
      raf = requestAnimationFrame(tick)
    }
  }

  const finish = () => {
    finished = true
    stop()
    scene.draw(Infinity)
    setShow(true)
  }

  const start = () => {
    started = true
    schedule()
  }

  let io: IntersectionObserver | undefined
  let ro: ResizeObserver | undefined
  let offTheme: (() => void) | undefined
  const onVisibility = () => {
    if (document.hidden) stop()
    else if (visible && started && !finished) schedule()
  }

  const boot = () => {
    if (!alive) return
    scene.paint()
    if (!size()) return
    setShow(scene.showFirst || still)
    if (still) finish()
    else if (scene.showFirst) scene.draw(0)
    if (!still) {
      io = new IntersectionObserver((entries) => {
        const entry = entries[entries.length - 1]
        visible = Boolean(entry?.isIntersecting)
        if (!visible) {
          stop()
          return
        }
        if (!started && (entry.intersectionRatio >= 0.35 || scene.showFirst)) start()
        else if (started && !finished) schedule()
      }, { threshold: [0, 0.35] })
      io.observe(host)
    }
    ro = new ResizeObserver(() => {
      if (Math.abs(host.clientWidth - w) < 1 && Math.abs(host.clientHeight - h) < 1) return
      if (!size()) return
      if (finished || still) scene.draw(Infinity)
      else if (scene.showFirst) scene.draw(0)
    })
    ro.observe(host)
    offTheme = onThemeChange(() => {
      scene.paint()
      if (finished || still) scene.draw(Infinity)
      else if (started || scene.showFirst) scene.draw(played)
    })
    document.addEventListener('visibilitychange', onVisibility)
  }

  // The dawn line is sampled from the rendered serif, so wait for the font.
  const fonts = (document as Document & { fonts?: FontFaceSet }).fonts
  let booted = false
  let wait = 0
  const bootOnce = () => {
    if (booted) return
    booted = true
    window.clearTimeout(wait)
    boot()
  }
  wait = window.setTimeout(bootOnce, 1200)
  if (fonts?.ready) void fonts.ready.then(bootOnce)
  else bootOnce()

  return () => {
    alive = false
    window.clearTimeout(wait)
    stop()
    io?.disconnect()
    ro?.disconnect()
    offTheme?.()
    document.removeEventListener('visibilitychange', onVisibility)
  }
}
