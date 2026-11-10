/* The masthead's field of blocks, ported from System Duo's hero (app.vllm-sr.ai, assets/hero.js): squares 6 px wide
 * and 2 px apart, in tiles of 8, 16, 32 and 64 px. It opens as coarse noise and denoises into one horizon that rises
 * from both edges to the centre, under the actions. The fast side snaps in four steps into crisp terraces of 16 px
 * ember tiles; the slow side refines over thirty-six, its blocks splitting from 64 px down to 8, into a fine blue
 * ridge. Under the actions the two interlock tile by tile and the crest runs through the colours of the "Duo"
 * gradient. Nothing is drawn above the actions ([data-field-calm]). Resolved, the crest shimmers and a fine pointer
 * stirs the blocks under it. Decoration only: drawn once and still under prefers-reduced-motion, paused off screen
 * and while the tab is hidden. */
import { numberVar, onThemeChange, reducedMotion, rgbVar, rng, smooth } from './pixels'

const K = 2
const U = 4
const CELL = K * U
const FAST = [0.62, 0.32, 0.1, 0]
const FAST_DT = 0.16
const SLOW = 1 / 36
const SLOW_DT = 0.1
const HOLD = 0.35
const SPLIT: Record<number, number> = { 8: 0.75, 4: 0.5, 2: 0.25 }
const RISE = 0.26
const ALPHA = [0.07, 0.11, 0.16, 0.23, 0.32, 0.44, 0.6, 0.8, 1]
const NL = ALPHA.length
const HUES = 3
const ACC = 1 + HUES * NL
const GRAD = 9
const MEET = ACC + 2
const BAYER4 = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
const BAYER8 = [0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38,
  60, 28, 52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25,
  15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21]

function noise1(seed: number, n: number): (x: number) => number {
  const r = rng(seed)
  const v = Array.from({ length: n + 1 }, r)
  return (x) => {
    const p = Math.max(0, Math.min(n - 0.0001, x))
    const i = Math.floor(p)
    const f = p - i
    const s = f * f * (3 - 2 * f)
    return v[i] * (1 - s) + v[i + 1] * s
  }
}

function fractal(seed: number): (x: number) => number {
  const a = noise1(seed, 64)
  const b = noise1(seed + 1, 128)
  const c = noise1(seed + 2, 256)
  return x => a(x * 5) * 0.62 + b(x * 13) * 0.26 + c(x * 34) * 0.12
}

type Summary = { cols: number, side: Uint8Array, empty: Uint8Array, avg: Float32Array, hue: Uint8Array, excl: Uint8Array }

export function mountField(wrap: HTMLElement): () => void {
  const canvas = wrap.querySelector('canvas')
  const ctx = canvas?.getContext('2d')
  const host = (wrap.closest('[data-field-host]') ?? wrap.parentElement) as HTMLElement | null
  if (!canvas || !ctx || !host) return () => undefined

  const reduce = reducedMotion()
  const fine = window.matchMedia('(pointer: fine)')
  const rand = Math.random
  let styles: string[] = []
  const buckets: number[][] = Array.from({ length: MEET + GRAD }, () => [])
  let cols = 0, rows = 0, n = 0, tc = 0, tr = 0, w = 0, h = 0, hh = 0
  let lv = new Uint8Array(0), hu = lv, two = lv, live = lv, dirty = lv, blank = lv
  let amp = new Float32Array(0)
  let sig = new Float64Array(0), next = sig
  let nodes: Record<number, Summary> = {}
  let edges: number[] = [], active: number[] = []
  let clock = 0, last = 0, raf = 0, idle = 0, lastCell = -1
  let visible = true, started = false, disposed = false

  const paint = () => {
    const boost = numberVar('--px-boost', 1)
    const hues = ['--px-ember', '--px-blue', '--px-violet'].map(name => rgbVar(name, '22 24 29'))
    styles = ['']
    for (const rgb of hues) ALPHA.forEach(a => styles.push(`rgba(${rgb}, ${Math.min(1, a * boost).toFixed(3)})`))
    styles.push(`rgb(${rgbVar('--px-ember-accent', '239 86 48')})`, `rgb(${rgbVar('--px-blue-accent', '91 130 224')})`)
    const stops = [1, 2, 3, 4].map(k => rgbVar(`--px-duo-${k}`, '120 70 190').split(', ').map(Number))
    for (let g = 0; g < GRAD; g++) {
      const t = g / (GRAD - 1) * (stops.length - 1)
      const g0 = Math.min(stops.length - 2, Math.floor(t))
      const f = t - g0
      styles.push(`rgb(${stops[g0].map((v, j) => Math.round(v + (stops[g0 + 1][j] - v) * f)).join(', ')})`)
    }
  }

  const build = (): boolean => {
    w = wrap.clientWidth
    h = wrap.clientHeight
    hh = host.clientHeight
    if (!w || !h) return false
    cols = Math.ceil(w / CELL / 8) * 8
    rows = Math.ceil(h / CELL / 8) * 8
    n = cols * rows
    tc = cols / 8
    tr = rows / 8
    canvas.width = cols * U
    canvas.height = rows * U
    canvas.style.width = `${cols * CELL}px`
    canvas.style.height = `${rows * CELL}px`
    lv = new Uint8Array(n)
    hu = new Uint8Array(n)
    two = new Uint8Array(n)
    amp = new Float32Array(n)
    sig = new Float64Array(n)
    next = new Float64Array(n)
    live = new Uint8Array(n)
    dirty = new Uint8Array(tc * tr)
    blank = new Uint8Array(tc * tr)
    edges = []
    active = []

    // The calm zone: no block, not even noise, above the actions. The canvas is anchored at the host's bottom.
    const calmBox = host.querySelector('[data-field-calm]')
    const hostTop = host.getBoundingClientRect().top
    const above = rows * CELL - hh
    let calm = calmBox ? Math.ceil((above + calmBox.getBoundingClientRect().bottom - hostTop + 28) / CELL) : 0
    calm = Math.max(0, Math.min(rows - 8, calm + (calm & 1)))
    const avail = rows - calm
    const r = rng(cols * 7 + 3)
    const X = (col: number) => (col + 0.5) * CELL / w
    const centre = (x: number) => Math.exp(-(((x - 0.5) / RISE) ** 2))
    const band = Math.max(0.14, 80 / w)
    const side = (x: number) => smooth(0.5 - band, 0.5 + band, x)
    const near = (x: number) => 1 - smooth(band * 0.5, band * 1.2, Math.abs(x - 0.5))
    const duo = (x: number) => MEET + Math.round(side(x) * (GRAD - 1))
    const body = (du: number, crest: number, x: number, yb: number) =>
      Math.exp(-du / Math.max(3, crest * 0.32)) * (0.7 + 0.3 * centre(x)) * smooth(-1, avail * 0.14, yb)

    for (let row = 0; row < rows; row++) {
      for (let c = 0; c < cols; c++) {
        const yb = rows - 1 - row
        amp[row * cols + c] = row < calm ? 0 : 0.1 + 0.9 * smooth(calm, calm + 6, row) * (0.3 + 0.7 * smooth(0, avail * 0.6, yb))
      }
    }
    // Each 16 px tile belongs to one side; where they meet an ordered dither interlocks them.
    const tcols = cols / 2
    const trows = rows / 2
    for (let ty = 0; ty < trows; ty++) {
      for (let tx = 0; tx < tcols; tx++) {
        const t2 = side(X(tx * 2 + 0.5)) > (BAYER4[(ty & 3) * 4 + (tx & 3)] + 0.5) / 16 ? 1 : 0
        for (let q = 0; q < 4; q++) two[(ty * 2 + (q >> 1)) * cols + tx * 2 + (q & 1)] = t2
      }
    }
    // One horizon for both sides: low at the edges, rising to the centre.
    const land = fractal(11)
    const grain = fractal(23)
    const horizon = (x: number) => {
      const e = centre(x)
      return Math.min(0.84, 0.4 + 0.32 * e + 0.46 * (land(x) - 0.5) * (1 - 0.7 * e))
    }
    // The fast side's terraces: runs of a few 16 px tiles at one height.
    const terr = new Int16Array(tcols)
    const lift = new Uint8Array(tcols)
    let prev = 0
    for (let tx = 0; tx < tcols;) {
      const run = 2 + ((r() * 4) | 0)
      const ht = Math.max(2, Math.round(avail / 2 * horizon(X(2 * (tx + run / 2))) + (r() - 0.5) * 1.4))
      for (let k = 0; k < run && tx < tcols; k++, tx++) {
        terr[tx] = ht
        lift[tx] = k === 0 && ht > prev ? 1 : 0
      }
      prev = ht
    }
    // The slow side: a fine ridge at unit resolution, its crest two blocks deep, a faint glow above it.
    for (let c = 0; c < cols; c++) {
      const x = X(c)
      const nr = near(x)
      const crest = Math.max(3, Math.round(avail * (horizon(x) + 0.07 * (grain(x * 2.6) - 0.5)) * (1 - nr) + terr[c >> 1] * 2 * nr))
      for (let yb = 0; yb < rows; yb++) {
        const row = rows - 1 - yb
        const i = row * cols + c
        if (!two[i] || !amp[i]) continue
        const d = crest - 1 - yb
        const th = (BAYER8[(row & 7) * 8 + (c & 7)] + 0.5) / 64
        let l = 0
        let hue = 1
        if (d >= 0) {
          const v = d < 2 ? 1 : body(d - 1, crest, x, yb)
          if (d < 2 || v > th * 0.75 + r() * 0.25) {
            if (d < 2 && nr > 0.05) l = duo(x)
            else if (d === 0) l = NL
            else l = d === 1 ? NL - 1 : Math.max(1, Math.round(v * 7))
          }
        }
        else if (d >= -4 && 0.42 * (1 - (-d - 1) / 4) ** 1.6 * (0.4 + 0.6 * centre(x)) > th) {
          l = r() < 0.08 ? ACC + 1 : 2
          hue = nr > 0.3 ? 2 : 1
        }
        lv[i] = l
        hu[i] = hue
        if (l && d <= 0) edges.push(i)
      }
    }
    // The fast side: an ember crest on each terrace, an accent where one rises, thinning out downwards.
    for (let tx = 0; tx < tcols; tx++) {
      const c0 = tx * 2
      const x0 = X(c0 + 0.5)
      for (let ty = 0; ty < trows; ty++) {
        const r0 = rows - 2 - ty * 2
        let ok = true
        for (let q = 0; q < 4; q++) {
          const i = (r0 + (q >> 1)) * cols + c0 + (q & 1)
          if (two[i] || !amp[i]) ok = false
        }
        if (!ok) continue
        const d = terr[tx] - 1 - ty
        const th = (BAYER4[(ty & 3) * 4 + (tx & 3)] + 0.5) / 16
        let l = 0
        if (d >= 0) {
          const v = d < 1 ? 1 : body(2 * d - 1, terr[tx] * 2, x0, ty * 2)
          if (d < 1 || v > th * 0.75 + r() * 0.25) {
            if (d) l = Math.max(1, Math.round(v * 7))
            else if (near(x0) > 0.05) l = duo(x0)
            else l = lift[tx] && r() < 0.6 ? ACC : NL
          }
        }
        for (let q = 0; q < 4; q++) {
          const i = (r0 + (q >> 1)) * cols + c0 + (q & 1)
          lv[i] = l
          hu[i] = 0
        }
        if (l && d === 0) edges.push(r0 * cols + c0)
      }
    }
    nodes = {}
    for (const S of [8, 4, 2]) nodes[S] = summarize(S)
    for (let t = 0; t < tc * tr; t++) blank[t] = nodes[8].excl[t] === 2 ? 1 : 0
    return true
  }

  // Per block of S units: its side (0 fast, 1 slow, 2 both), whether it is empty, its mean level and main hue, and
  // whether it lies in the calm zone (2: all of it).
  function summarize(S: number): Summary {
    const nc = cols / S
    const m = nc * (rows / S)
    const out: Summary = {
      cols: nc,
      side: new Uint8Array(m),
      empty: new Uint8Array(m),
      avg: new Float32Array(m),
      hue: new Uint8Array(m),
      excl: new Uint8Array(m),
    }
    for (let k = 0; k < m; k++) {
      const ux = (k % nc) * S
      const uy = ((k / nc) | 0) * S
      let sum = 0, outside = 0, ones = 0
      const cnt = [0, 0, 0]
      for (let y = uy; y < uy + S; y++) {
        for (let x = ux; x < ux + S; x++) {
          const i = y * cols + x
          const l = lv[i]
          sum += l >= ACC ? 7 : l
          if (!amp[i]) outside++
          if (!two[i]) ones++
          if (l) cnt[hu[i]]++
        }
      }
      out.side[k] = ones === S * S ? 0 : ones ? 2 : 1
      out.empty[k] = sum ? 0 : 1
      out.avg[k] = sum / (S * S) / NL
      out.hue[k] = cnt[0] || cnt[1] || cnt[2] ? Number(cnt[0] < cnt[1]) : Number(ones * 2 < S * S)
      out.excl[k] = outside === S * S ? 2 : outside ? 1 : 0
    }
    return out
  }

  const style = (level: number, hue: number) => {
    const l = Math.min(NL, Math.round(level))
    return l < 1 ? 0 : 1 + hue * NL + l - 1
  }
  const settledStyle = (i: number) => (lv[i] >= ACC ? lv[i] : style(lv[i], hu[i]))
  const noisy = (mean: number, s: number, a: number) => {
    const z = rand()
    return ((1 - s) * mean + s * z * z * z * 0.9 * a) * NL
  }
  // Coarse noise leaves its faintest blocks out, so even the first steps are blocks and spacing, not a grid.
  const coarse = (mean: number, s: number, a: number) => {
    const v = noisy(mean, s, a)
    return v < 1.6 ? 0 : v
  }
  const block = (ux: number, uy: number, S: number, p: number) => {
    if (p) buckets[p].push(ux * U, uy * U, S * U - 1)
  }

  const drawNode = (S: number, ux: number, uy: number) => {
    const c = uy * cols + ux
    if (S === 1) {
      if (!amp[c]) return
      if (!sig[c]) block(ux, uy, 1, settledStyle(c))
      else if (sig[c] < 0.12 && lv[c] >= ACC) block(ux, uy, 1, lv[c])
      else block(ux, uy, 1, style(noisy((lv[c] >= ACC ? 7 : lv[c]) / NL, sig[c], amp[c]), hu[c]))
      return
    }
    const N = nodes[S]
    const k = (uy / S) * N.cols + ux / S
    if (N.excl[k] === 2) return
    let lo = 2, hi = 0, sum = 0, a = 0
    for (let y = uy; y < uy + S; y++) {
      for (let x = ux; x < ux + S; x++) {
        const i = y * cols + x
        lo = Math.min(lo, sig[i])
        hi = Math.max(hi, sig[i])
        sum += sig[i]
        a += amp[i]
      }
    }
    if (!hi && N.empty[k]) return
    const sm = sum / (S * S)
    const am = a / (S * S)
    if (!N.excl[k] && lo >= SPLIT[S]) {
      block(ux, uy, S, style(coarse(N.avg[k], sm, am), N.hue[k]))
      return
    }
    // The fast side's 16 px tiles never split further: they settle whole.
    if (S === 2 && N.side[k] === 0) {
      if (N.excl[k]) return
      if (!hi) block(ux, uy, 2, settledStyle(c))
      else block(ux, uy, 2, sm < 0.12 && lv[c] >= ACC ? lv[c] : style(coarse(N.avg[k], sm, am), N.hue[k]))
      return
    }
    const half = S / 2
    drawNode(half, ux, uy)
    drawNode(half, ux + half, uy)
    drawNode(half, ux, uy + half)
    drawNode(half, ux + half, uy + half)
  }

  const redraw = (all: boolean) => {
    for (let t = 0; t < tc * tr; t++) {
      if (!all && !dirty[t]) continue
      dirty[t] = 0
      if (blank[t]) continue
      const ux = (t % tc) * 8
      const uy = ((t / tc) | 0) * 8
      ctx.clearRect(ux * U, uy * U, 8 * U, 8 * U)
      drawNode(8, ux, uy)
    }
    for (let p = 1; p < buckets.length; p++) {
      const b = buckets[p]
      if (!b.length) continue
      ctx.fillStyle = styles[p]
      for (let q = 0; q < b.length; q += 3) ctx.fillRect(b[q], b[q + 1], b[q + 2], b[q + 2])
      b.length = 0
    }
  }

  const tileOf = (i: number) => (((i / cols) | 0) >> 3) * tc + ((i % cols) >> 3)
  const stir = (i: number, s: number) => {
    if (!amp[i] || s <= sig[i]) return
    sig[i] = s
    next[i] = clock
    dirty[tileOf(i)] = 1
    if (!live[i]) {
      live[i] = 1
      active.push(i)
    }
  }
  // The fast side's tiles move as one: stirring a cell stirs its whole tile.
  const stirAt = (i: number, s: number) => {
    if (two[i]) {
      stir(i, s)
      return
    }
    const r0 = ((i / cols) | 0) & ~1
    const c0 = (i % cols) & ~1
    for (let q = 0; q < 4; q++) stir((r0 + (q >> 1)) * cols + c0 + (q & 1), s)
  }

  const settle = () => {
    sig.fill(0)
    live.fill(0)
    active = []
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    redraw(true)
    wrap.dataset.state = 'on'
  }
  const intro = () => {
    active = []
    for (let i = 0; i < n; i++) {
      if (!amp[i]) continue
      sig[i] = 1
      live[i] = 1
      active.push(i)
      next[i] = clock + HOLD + (two[i] ? SLOW_DT : FAST_DT)
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    redraw(true)
    wrap.dataset.state = 'on'
  }
  const step = (now: number) => {
    const keep: number[] = []
    for (const i of active) {
      if (next[i] <= now) {
        let s = sig[i]
        if (two[i]) {
          s = s - SLOW > 0.001 ? s - SLOW : 0
          next[i] = now + SLOW_DT
        }
        else {
          s = FAST.find(f => s > f + 1e-6) ?? 0
          next[i] = now + FAST_DT
        }
        sig[i] = s
        dirty[tileOf(i)] = 1
      }
      if (sig[i] > 0) keep.push(i)
      else live[i] = 0
    }
    active = keep
  }
  // Resolved, a few blocks along the crest and in the glow catch a little noise and settle again.
  const shimmer = (dt: number) => {
    let rate = (w < 700 ? 5 : 12) * dt
    while (edges.length && rate > 0) {
      if (rate >= 1 || rand() < rate) stirAt(edges[(rand() * edges.length) | 0], 0.3)
      rate -= 1
    }
  }

  const running = () => !disposed && visible && !document.hidden && !reduce.matches
  const schedule = () => {
    if (raf || idle || disposed) return
    if (active.length) raf = window.requestAnimationFrame(frame)
    else idle = window.setTimeout(tick, 1000 / 12)
  }
  function frame(t: number) {
    raf = 0
    if (!running()) {
      last = 0
      return
    }
    const dt = last ? Math.min(0.05, (t - last) / 1000) : 0
    last = t
    clock += dt
    shimmer(dt)
    step(clock)
    redraw(false)
    schedule()
  }
  // Resolved, the field needs no frames: a slow tick only seeds the shimmer, and frames run while it settles.
  function tick() {
    idle = 0
    if (!running()) return
    clock += 1 / 12
    shimmer(1 / 12)
    if (active.length) last = 0
    schedule()
  }
  const start = () => {
    if (disposed || (!n && !build())) return
    if (reduce.matches) {
      settle()
      return
    }
    if (!started) {
      started = true
      intro()
    }
    if (running()) schedule()
  }

  const onPointer = (event: PointerEvent) => {
    if (!fine.matches || !running() || !n) return
    const box = canvas.getBoundingClientRect()
    const pc = Math.floor((event.clientX - box.left) / CELL)
    const pr = Math.floor((event.clientY - box.top) / CELL)
    if (pr < 0 || pr >= rows || pc < 0 || pc >= cols || pr * cols + pc === lastCell) return
    lastCell = pr * cols + pc
    const R = 6
    for (let row = Math.max(0, pr - R); row <= Math.min(rows - 1, pr + R); row++) {
      for (let c = Math.max(0, pc - R); c <= Math.min(cols - 1, pc + R); c++) {
        const d = Math.hypot(row - pr, c - pc)
        if (d < R) stirAt(row * cols + c, 0.9 * (1 - d / R) ** 0.7)
      }
    }
    if (idle) {
      window.clearTimeout(idle)
      idle = 0
    }
    schedule()
  }
  const onVisibility = () => {
    if (!document.hidden) start()
  }

  let seen = 0
  const resizeObserver = new ResizeObserver(() => {
    if (!seen++ || !n) return
    if (wrap.clientWidth === w && Math.abs(wrap.clientHeight - h) < 24 && Math.abs(host.clientHeight - hh) < 24) return
    if (build()) settle()
  })
  const intersectionObserver = new IntersectionObserver((entries) => {
    visible = entries[0]?.isIntersecting ?? true
    if (visible) start()
  })
  const offTheme = onThemeChange(() => {
    paint()
    if (n) redraw(true)
  })

  paint()
  // Measure the actions in their final font, or the calm zone would follow the fallback's metrics.
  const fontsReady = Promise.race([document.fonts?.ready, new Promise(done => window.setTimeout(done, 1200))])
  fontsReady.then(() => {
    if (disposed) return
    resizeObserver.observe(host)
    intersectionObserver.observe(wrap)
    host.addEventListener('pointermove', onPointer, { passive: true })
    document.addEventListener('visibilitychange', onVisibility)
    reduce.addEventListener('change', start)
    start()
  })

  return () => {
    disposed = true
    if (raf) window.cancelAnimationFrame(raf)
    if (idle) window.clearTimeout(idle)
    resizeObserver.disconnect()
    intersectionObserver.disconnect()
    offTheme()
    host.removeEventListener('pointermove', onPointer)
    document.removeEventListener('visibilitychange', onVisibility)
    reduce.removeEventListener('change', start)
  }
}
