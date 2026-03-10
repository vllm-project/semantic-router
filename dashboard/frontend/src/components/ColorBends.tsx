import { useEffect, useRef } from 'react'
import type { CSSProperties } from 'react'
import * as THREE from 'three'
import './ColorBends.css'

type ColorBendsProps = {
  className?: string
  style?: CSSProperties
  rotation?: number
  speed?: number
  colors?: string[]
  transparent?: boolean
  autoRotate?: number
  scale?: number
  frequency?: number
  warpStrength?: number
  mouseInfluence?: number
  parallax?: number
  noise?: number
}

const MAX_COLORS = 8 as const

const frag = `
#define MAX_COLORS ${MAX_COLORS}
uniform vec2 uCanvas;
uniform float uTime;
uniform float uSpeed;
uniform vec2 uRot;
uniform int uColorCount;
uniform vec3 uColors[MAX_COLORS];
uniform int uTransparent;
uniform float uScale;
uniform float uFrequency;
uniform float uWarpStrength;
uniform vec2 uPointer;
uniform float uMouseInfluence;
uniform float uParallax;
uniform float uNoise;
varying vec2 vUv;

void main() {
  float t = uTime * uSpeed;
  vec2 p = vUv * 2.0 - 1.0;
  p += uPointer * uParallax * 0.1;
  vec2 rp = vec2(p.x * uRot.x - p.y * uRot.y, p.x * uRot.y + p.y * uRot.x);
  vec2 q = vec2(rp.x * (uCanvas.x / uCanvas.y), rp.y);
  q /= max(uScale, 0.0001);
  q /= 0.5 + 0.2 * dot(q, q);
  q += 0.2 * cos(t) - 7.56;
  vec2 toward = (uPointer - rp);
  q += toward * uMouseInfluence * 0.2;

  vec3 col = vec3(0.0);
  float a = 1.0;

  if (uColorCount > 0) {
    vec2 s = q;
    vec3 sumCol = vec3(0.0);
    float cover = 0.0;
    for (int i = 0; i < MAX_COLORS; ++i) {
      if (i >= uColorCount) break;
      s -= 0.01;
      vec2 r = sin(1.5 * (s.yx * uFrequency) + 2.0 * cos(s * uFrequency));
      float m0 = length(r + sin(5.0 * r.y * uFrequency - 3.0 * t + float(i)) / 4.0);
      float kBelow = clamp(uWarpStrength, 0.0, 1.0);
      float kMix = pow(kBelow, 0.3);
      float gain = 1.0 + max(uWarpStrength - 1.0, 0.0);
      vec2 disp = (r - s) * kBelow;
      vec2 warped = s + disp * gain;
      float m1 = length(warped + sin(5.0 * warped.y * uFrequency - 3.0 * t + float(i)) / 4.0);
      float m = mix(m0, m1, kMix);
      float w = 1.0 - exp(-6.0 / exp(6.0 * m));
      sumCol += uColors[i] * w;
      cover = max(cover, w);
    }
    col = clamp(sumCol, 0.0, 1.0);
    a = uTransparent > 0 ? cover : 1.0;
  } else {
    vec2 s = q;
    for (int k = 0; k < 3; ++k) {
      s -= 0.01;
      vec2 r = sin(1.5 * (s.yx * uFrequency) + 2.0 * cos(s * uFrequency));
      float m0 = length(r + sin(5.0 * r.y * uFrequency - 3.0 * t + float(k)) / 4.0);
      float kBelow = clamp(uWarpStrength, 0.0, 1.0);
      float kMix = pow(kBelow, 0.3);
      float gain = 1.0 + max(uWarpStrength - 1.0, 0.0);
      vec2 disp = (r - s) * kBelow;
      vec2 warped = s + disp * gain;
      float m1 = length(warped + sin(5.0 * warped.y * uFrequency - 3.0 * t + float(k)) / 4.0);
      float m = mix(m0, m1, kMix);
      col[k] = 1.0 - exp(-6.0 / exp(6.0 * m));
    }
    a = uTransparent > 0 ? max(max(col.r, col.g), col.b) : 1.0;
  }

  if (uNoise > 0.0001) {
    float n = fract(sin(dot(gl_FragCoord.xy + vec2(uTime), vec2(12.9898, 78.233))) * 43758.5453123);
    col += (n - 0.5) * uNoise;
    col = clamp(col, 0.0, 1.0);
  }

  vec3 rgb = (uTransparent > 0) ? col * a : col;
  gl_FragColor = vec4(rgb, a);
}
`

const vert = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 1.0);
}
`

const hexToVector3 = (hex: string): THREE.Vector3 => {
  const clean = hex.replace('#', '').trim()
  if (!clean) return new THREE.Vector3(0, 0, 0)

  const value =
    clean.length === 3
      ? `${clean[0]}${clean[0]}${clean[1]}${clean[1]}${clean[2]}${clean[2]}`
      : clean.slice(0, 6)

  const parsed = Number.parseInt(value, 16)
  if (Number.isNaN(parsed)) return new THREE.Vector3(0, 0, 0)

  return new THREE.Vector3(
    ((parsed >> 16) & 255) / 255,
    ((parsed >> 8) & 255) / 255,
    (parsed & 255) / 255
  )
}

export default function ColorBends({
  className,
  style,
  rotation = 45,
  speed = 0.2,
  colors = [],
  transparent = true,
  autoRotate = 0,
  scale = 1,
  frequency = 1,
  warpStrength = 1,
  mouseInfluence = 1,
  parallax = 0.5,
  noise = 0.1,
}: ColorBendsProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const materialRef = useRef<THREE.ShaderMaterial | null>(null)
  const rafRef = useRef<number | null>(null)
  const resizeObserverRef = useRef<ResizeObserver | null>(null)
  const pointerTargetRef = useRef(new THREE.Vector2(0, 0))
  const pointerCurrentRef = useRef(new THREE.Vector2(0, 0))
  const rotationRef = useRef(rotation)
  const autoRotateRef = useRef(autoRotate)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const scene = new THREE.Scene()
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    const geometry = new THREE.PlaneGeometry(2, 2)

    const uColorsArray: THREE.Vector3[] = Array.from({ length: MAX_COLORS }, () => new THREE.Vector3())

    const material = new THREE.ShaderMaterial({
      vertexShader: vert,
      fragmentShader: frag,
      uniforms: {
        uCanvas: { value: new THREE.Vector2(1, 1) },
        uTime: { value: 0 },
        uSpeed: { value: speed },
        uRot: { value: new THREE.Vector2(1, 0) },
        uColorCount: { value: 0 },
        uColors: { value: uColorsArray },
        uTransparent: { value: transparent ? 1 : 0 },
        uScale: { value: scale },
        uFrequency: { value: frequency },
        uWarpStrength: { value: warpStrength },
        uPointer: { value: new THREE.Vector2(0, 0) },
        uMouseInfluence: { value: mouseInfluence },
        uParallax: { value: parallax },
        uNoise: { value: noise },
      },
      premultipliedAlpha: true,
      transparent: true,
    })

    materialRef.current = material

    const mesh = new THREE.Mesh(geometry, material)
    scene.add(mesh)

    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      powerPreference: 'high-performance',
      alpha: true,
    })

    rendererRef.current = renderer
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setClearColor(0x000000, transparent ? 0 : 1)
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'
    container.appendChild(renderer.domElement)

    const handleResize = () => {
      const width = container.clientWidth || 1
      const height = container.clientHeight || 1
      renderer.setSize(width, height, false)
      ;(material.uniforms.uCanvas.value as THREE.Vector2).set(width, height)
    }

    handleResize()

    let usingWindowResize = false

    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(handleResize)
      observer.observe(container)
      resizeObserverRef.current = observer
    } else {
      usingWindowResize = true
      window.addEventListener('resize', handleResize)
    }

    const clock = new THREE.Clock()

    const loop = () => {
      const delta = clock.getDelta()
      const elapsed = clock.elapsedTime
      material.uniforms.uTime.value = elapsed

      const degree = (rotationRef.current % 360) + autoRotateRef.current * elapsed
      const radian = (degree * Math.PI) / 180
      ;(material.uniforms.uRot.value as THREE.Vector2).set(Math.cos(radian), Math.sin(radian))

      const currentPointer = pointerCurrentRef.current
      const targetPointer = pointerTargetRef.current
      const lerpAmount = Math.min(1, delta * 8)
      currentPointer.lerp(targetPointer, lerpAmount)
      ;(material.uniforms.uPointer.value as THREE.Vector2).copy(currentPointer)

      renderer.render(scene, camera)
      rafRef.current = requestAnimationFrame(loop)
    }

    rafRef.current = requestAnimationFrame(loop)

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect()
      }
      if (usingWindowResize) {
        window.removeEventListener('resize', handleResize)
      }

      geometry.dispose()
      material.dispose()
      renderer.dispose()

      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  useEffect(() => {
    const material = materialRef.current
    const renderer = rendererRef.current
    if (!material) return

    rotationRef.current = rotation
    autoRotateRef.current = autoRotate
    material.uniforms.uSpeed.value = speed
    material.uniforms.uScale.value = scale
    material.uniforms.uFrequency.value = frequency
    material.uniforms.uWarpStrength.value = warpStrength
    material.uniforms.uMouseInfluence.value = mouseInfluence
    material.uniforms.uParallax.value = parallax
    material.uniforms.uNoise.value = noise

    const parsedColors = colors.filter(Boolean).slice(0, MAX_COLORS).map(hexToVector3)

    for (let index = 0; index < MAX_COLORS; index += 1) {
      const colorVec = (material.uniforms.uColors.value as THREE.Vector3[])[index]
      if (index < parsedColors.length) {
        colorVec.copy(parsedColors[index])
      } else {
        colorVec.set(0, 0, 0)
      }
    }

    material.uniforms.uColorCount.value = parsedColors.length
    material.uniforms.uTransparent.value = transparent ? 1 : 0

    if (renderer) {
      renderer.setClearColor(0x000000, transparent ? 0 : 1)
    }
  }, [
    autoRotate,
    colors,
    frequency,
    mouseInfluence,
    noise,
    parallax,
    rotation,
    scale,
    speed,
    transparent,
    warpStrength,
  ])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const handlePointerMove = (event: PointerEvent) => {
      const rect = container.getBoundingClientRect()
      const x = ((event.clientX - rect.left) / (rect.width || 1)) * 2 - 1
      const y = -(((event.clientY - rect.top) / (rect.height || 1)) * 2 - 1)
      pointerTargetRef.current.set(x, y)
    }

    const handlePointerLeave = () => {
      pointerTargetRef.current.set(0, 0)
    }

    container.addEventListener('pointermove', handlePointerMove)
    container.addEventListener('pointerleave', handlePointerLeave)

    return () => {
      container.removeEventListener('pointermove', handlePointerMove)
      container.removeEventListener('pointerleave', handlePointerLeave)
    }
  }, [])

  const containerClassName = className
    ? `color-bends-container ${className}`
    : 'color-bends-container'

  return <div ref={containerRef} className={containerClassName} style={style} />
}
