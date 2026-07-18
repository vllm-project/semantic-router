import React, { useEffect, useRef } from 'react'
import {
  findTerrainContourIndex,
  getTerrainElevation,
  takeWrappedText,
  TERRAIN_CONTOURS,
} from './terrain'
import styles from './index.module.css'

const FONT_SIZE = 10
const FONT_STACK
  = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace'
const FONT = `${FONT_SIZE}px ${FONT_STACK}`
const MIN_SEGMENT_WIDTH = 22

type ViewState = {
  panX: number
  panY: number
  targetPanX: number
  targetPanY: number
  targetZoom: number
  zoom: number
}

function drawTerrain(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  view: ViewState,
): void {
  const lineHeight = width <= 640 ? 13 : 14
  const scanStep = width >= 1800 ? 6 : width <= 640 ? 5 : 4
  const cursors = TERRAIN_CONTOURS.map(() => 0)

  context.fillStyle = '#030303'
  context.fillRect(0, 0, width, height)
  context.font = FONT
  context.textBaseline = 'top'

  const drawSegment = (
    contourIndex: number,
    x: number,
    y: number,
    availableWidth: number,
  ) => {
    if (availableWidth <= MIN_SEGMENT_WIDTH) return

    const contour = TERRAIN_CONTOURS[contourIndex]
    const characterCount = Math.max(
      1,
      Math.floor(availableWidth / (FONT_SIZE * 0.61)),
    )
    const next = takeWrappedText(
      contour.text,
      cursors[contourIndex],
      characterCount,
    )
    cursors[contourIndex] = next.cursor
    context.fillStyle = contour.color
    context.fillText(next.text, x, y, availableWidth)
  }

  for (let y = 0; y < height; y += lineHeight) {
    const mapY = (y - view.panY) / view.zoom
    let activeContourIndex = -1
    let segmentStartX = 0

    for (let x = 0; x <= width; x += scanStep) {
      const mapX = (x - view.panX) / view.zoom
      const contourIndex = findTerrainContourIndex(
        getTerrainElevation(mapX, mapY),
      )

      if (contourIndex === activeContourIndex) continue

      if (activeContourIndex !== -1) {
        drawSegment(
          activeContourIndex,
          segmentStartX,
          y,
          x - segmentStartX,
        )
      }
      activeContourIndex = contourIndex
      segmentStartX = x
    }

    if (activeContourIndex !== -1) {
      drawSegment(
        activeContourIndex,
        segmentStartX,
        y,
        width - segmentStartX,
      )
    }
  }
}

export default function TerrainCanvas(): JSX.Element {
  const rootRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const latitudeRef = useRef<HTMLSpanElement>(null)
  const longitudeRef = useRef<HTMLSpanElement>(null)
  const scaleRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    const root = rootRef.current
    const canvas = canvasRef.current
    if (!root || !canvas) return undefined

    const context = canvas.getContext('2d', { alpha: false })
    if (!context) return undefined

    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    const view: ViewState = {
      panX: 0,
      panY: 0,
      targetPanX: 0,
      targetPanY: 0,
      zoom: 1,
      targetZoom: 1,
    }
    let width = 0
    let height = 0
    let frame = 0
    let isVisible = true
    let isDragging = false
    let activePointerId: number | null = null
    let lastPointerX = 0
    let lastPointerY = 0

    const updateHud = () => {
      const centerMapX = (width / 2 - view.panX) / view.zoom
      const centerMapY = (height / 2 - view.panY) / view.zoom
      if (latitudeRef.current) {
        latitudeRef.current.textContent = centerMapY.toFixed(4)
      }
      if (longitudeRef.current) {
        longitudeRef.current.textContent = centerMapX.toFixed(4)
      }
      if (scaleRef.current) {
        scaleRef.current.textContent = `${view.zoom.toFixed(3)}X`
      }
    }

    const paint = () => {
      frame = 0
      if (!isVisible || document.visibilityState === 'hidden') return

      const reducedMotion = reducedMotionQuery.matches
      if (reducedMotion) {
        view.panX = view.targetPanX
        view.panY = view.targetPanY
        view.zoom = view.targetZoom
      }
      else {
        view.panX += (view.targetPanX - view.panX) * 0.15
        view.panY += (view.targetPanY - view.panY) * 0.15
        view.zoom += (view.targetZoom - view.zoom) * 0.15
      }

      drawTerrain(context, width, height, view)
      updateHud()

      const stillMoving
        = Math.abs(view.targetPanX - view.panX) > 0.08
          || Math.abs(view.targetPanY - view.panY) > 0.08
          || Math.abs(view.targetZoom - view.zoom) > 0.0008
      if (stillMoving && !reducedMotion) {
        frame = window.requestAnimationFrame(paint)
      }
    }

    const requestPaint = () => {
      if (!frame && isVisible) frame = window.requestAnimationFrame(paint)
    }

    const resize = () => {
      const rect = root.getBoundingClientRect()
      width = Math.max(1, Math.round(rect.width))
      height = Math.max(1, Math.round(rect.height))
      const devicePixelRatio = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = Math.round(width * devicePixelRatio)
      canvas.height = Math.round(height * devicePixelRatio)
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0)
      requestPaint()
    }

    const onPointerDown = (event: PointerEvent) => {
      if (event.pointerType === 'touch' || event.button !== 0) return
      isDragging = true
      activePointerId = event.pointerId
      lastPointerX = event.clientX
      lastPointerY = event.clientY
      canvas.setPointerCapture(event.pointerId)
      canvas.dataset.dragging = 'true'
    }

    const onPointerMove = (event: PointerEvent) => {
      if (!isDragging || event.pointerId !== activePointerId) return
      view.targetPanX += event.clientX - lastPointerX
      view.targetPanY += event.clientY - lastPointerY
      lastPointerX = event.clientX
      lastPointerY = event.clientY
      requestPaint()
    }

    const finishPointer = (event: PointerEvent) => {
      if (event.pointerId !== activePointerId) return
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId)
      }
      isDragging = false
      activePointerId = null
      delete canvas.dataset.dragging
    }

    const onWheel = (event: WheelEvent) => {
      if (!event.altKey) return
      event.preventDefault()
      const zoomDelta = -event.deltaY * 0.002
      const nextZoom = Math.max(
        0.35,
        Math.min(view.targetZoom * (1 + zoomDelta), 5),
      )
      const rect = canvas.getBoundingClientRect()
      const pointerX = event.clientX - rect.left
      const pointerY = event.clientY - rect.top
      const mapX = (pointerX - view.targetPanX) / view.targetZoom
      const mapY = (pointerY - view.targetPanY) / view.targetZoom
      view.targetZoom = nextZoom
      view.targetPanX = pointerX - mapX * nextZoom
      view.targetPanY = pointerY - mapY * nextZoom
      requestPaint()
    }

    const onDoubleClick = () => {
      view.targetPanX = 0
      view.targetPanY = 0
      view.targetZoom = 1
      requestPaint()
    }

    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') requestPaint()
    }

    const resizeObserver = new ResizeObserver(resize)
    const intersectionObserver = new IntersectionObserver((entries) => {
      isVisible = entries[0]?.isIntersecting ?? true
      if (isVisible) requestPaint()
      else if (frame) {
        window.cancelAnimationFrame(frame)
        frame = 0
      }
    })

    resizeObserver.observe(root)
    intersectionObserver.observe(root)
    canvas.addEventListener('pointerdown', onPointerDown)
    canvas.addEventListener('pointermove', onPointerMove)
    canvas.addEventListener('pointerup', finishPointer)
    canvas.addEventListener('pointercancel', finishPointer)
    canvas.addEventListener('wheel', onWheel, { passive: false })
    canvas.addEventListener('dblclick', onDoubleClick)
    document.addEventListener('visibilitychange', onVisibilityChange)
    reducedMotionQuery.addEventListener('change', requestPaint)
    resize()

    return () => {
      if (frame) window.cancelAnimationFrame(frame)
      resizeObserver.disconnect()
      intersectionObserver.disconnect()
      canvas.removeEventListener('pointerdown', onPointerDown)
      canvas.removeEventListener('pointermove', onPointerMove)
      canvas.removeEventListener('pointerup', finishPointer)
      canvas.removeEventListener('pointercancel', finishPointer)
      canvas.removeEventListener('wheel', onWheel)
      canvas.removeEventListener('dblclick', onDoubleClick)
      document.removeEventListener('visibilitychange', onVisibilityChange)
      reducedMotionQuery.removeEventListener('change', requestPaint)
    }
  }, [])

  return (
    <div ref={rootRef} className={styles.terrain} aria-hidden="true">
      <canvas ref={canvasRef} className={styles.terrainCanvas} />

      <div className={styles.terrainHud}>
        <div>
          Latitude
          <span ref={latitudeRef}>0.0000</span>
        </div>
        <div>
          Longitude
          <span ref={longitudeRef}>0.0000</span>
        </div>
        <div>
          Zoom scale
          <span ref={scaleRef}>1.000X</span>
        </div>
      </div>

      <div className={styles.terrainCrosshair} />

      <div className={styles.terrainLegend}>
        {TERRAIN_CONTOURS.map(contour => (
          <div key={contour.label} className={styles.terrainLegendItem}>
            <span style={{ background: contour.color }} />
            <strong style={{ color: contour.color }}>{contour.label}</strong>
          </div>
        ))}
      </div>
    </div>
  )
}
