import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useSetup } from '../contexts/SetupContext'
import {
  AUTH_TRANSITION_MIN_DURATION_MS,
  sanitizeAuthTransitionTarget,
} from './authTransitionSupport'
import styles from './AuthTransitionPage.module.css'

type Milestone = {
  year: string
  text: string
  revealAt: number
}

const PROGRESS_SEGMENTS = [
  { duration: 700, from: 0, to: 14 },
  { duration: 650, from: 14, to: 45 },
  { duration: 850, from: 45, to: 45 },
  { duration: 950, from: 45, to: 82 },
  { duration: 650, from: 82, to: 100 },
]

const MILESTONES: Milestone[] = [
  {
    year: '1938',
    text: 'Claude Shannon - A Symbolic Analysis of Relay and Switching Circuits',
    revealAt: 8,
  },
  {
    year: '1948',
    text: 'Claude Shannon - A Mathematical Theory of Communication',
    revealAt: 32,
  },
  {
    year: '2025',
    text: 'vLLM Semantic Router: Signal Decision Routing for Mixture-of-Modalities Models',
    revealAt: 56,
  },
  {
    year: 'Future',
    text: 'Standing on the shoulders of giants, we honor that legacy and explore the future together.',
    revealAt: 78,
  },
]

function easeOutCubic(value: number): number {
  return 1 - (1 - value) ** 3
}

function getTransitionProgress(elapsedMs: number): number {
  let consumedDuration = 0

  for (const segment of PROGRESS_SEGMENTS) {
    const segmentEnd = consumedDuration + segment.duration
    if (elapsedMs <= segmentEnd) {
      const localProgress = (elapsedMs - consumedDuration) / segment.duration
      const easedProgress = easeOutCubic(Math.max(0, Math.min(localProgress, 1)))
      return segment.from + (segment.to - segment.from) * easedProgress
    }
    consumedDuration = segmentEnd
  }

  return 100
}

const TransitionScene: React.FC<{ progress: number }> = ({ progress }) => {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const progressRef = useRef(progress)

  useEffect(() => {
    progressRef.current = progress
  }, [progress])

  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    const scene = new THREE.Scene()
    const aspect = container.clientWidth / Math.max(container.clientHeight, 1)
    const viewDistance = 5
    const camera = new THREE.OrthographicCamera(
      -viewDistance * aspect,
      viewDistance * aspect,
      viewDistance,
      -viewDistance,
      1,
      1000,
    )
    camera.position.set(10, 10, 10)
    camera.lookAt(scene.position)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x000000, 0)
    container.appendChild(renderer.domElement)

    const noiseVertexShader = `
      varying vec2 vUv;

      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `

    const noiseFragmentShader = `
      uniform float uTime;
      uniform vec2 uResolution;
      varying vec2 vUv;

      float random(vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
      }

      void main() {
        vec2 center = vec2(0.45, 0.45);
        float dist = distance(vUv, center);
        float angle = atan(vUv.y - center.y, vUv.x - center.x);
        float radius = 0.30 + 0.08 * sin(angle * 3.0 + uTime * 0.55);
        float mask = smoothstep(radius, 0.0, dist);
        vec2 noiseUv = vUv * min(uResolution.x, uResolution.y) * 0.5;
        float noiseVal = random(noiseUv + uTime * 0.1);
        float stipple = step(noiseVal, mask * 1.4);
        gl_FragColor = vec4(vec3(1.0), stipple * 0.24);
      }
    `

    const noiseGeometry = new THREE.PlaneGeometry(15, 15)
    const noiseMaterial = new THREE.ShaderMaterial({
      vertexShader: noiseVertexShader,
      fragmentShader: noiseFragmentShader,
      transparent: true,
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2(container.clientWidth, container.clientHeight) },
      },
    })
    const noisePlane = new THREE.Mesh(noiseGeometry, noiseMaterial)
    noisePlane.lookAt(camera.position)
    noisePlane.position.set(-1, -1, -2)
    scene.add(noisePlane)

    const group = new THREE.Group()
    scene.add(group)

    const boxGeometry = new THREE.BoxGeometry(6, 6, 6)
    const boxEdges = new THREE.EdgesGeometry(boxGeometry)
    const wireframeMaterial = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 1,
    })
    const wireframeBox = new THREE.LineSegments(boxEdges, wireframeMaterial)
    group.add(wireframeBox)

    const cylinderGeometry = new THREE.CylinderGeometry(2.5, 2.5, 0.5, 32)
    const fillVertexShader = `
      varying vec3 vPosition;

      void main() {
        vPosition = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `

    const fillFragmentShader = `
      varying vec3 vPosition;
      uniform float uProgress;

      void main() {
        float wipe = step(vPosition.y + 2.5, uProgress * 5.0);
        if (wipe < 0.5) {
          discard;
        }
        gl_FragColor = vec4(vec3(1.0), 1.0);
      }
    `

    const fillMaterial = new THREE.ShaderMaterial({
      vertexShader: fillVertexShader,
      fragmentShader: fillFragmentShader,
      side: THREE.DoubleSide,
      uniforms: { uProgress: { value: 0 } },
    })
    const fillMesh = new THREE.Mesh(cylinderGeometry, fillMaterial)
    fillMesh.rotation.x = Math.PI / 2
    group.add(fillMesh)

    const cylinderEdges = new THREE.EdgesGeometry(cylinderGeometry)
    const cylinderWireframe = new THREE.LineSegments(
      cylinderEdges,
      new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.54,
      }),
    )
    fillMesh.add(cylinderWireframe)

    const clock = new THREE.Clock()
    let frameId = 0
    let currentProgress = 0

    const updateRendererSize = () => {
      const width = container.clientWidth || window.innerWidth
      const height = container.clientHeight || window.innerHeight
      const nextAspect = width / Math.max(height, 1)

      camera.left = -viewDistance * nextAspect
      camera.right = viewDistance * nextAspect
      camera.top = viewDistance
      camera.bottom = -viewDistance
      camera.updateProjectionMatrix()

      renderer.setSize(width, height)
      noiseMaterial.uniforms.uResolution.value.set(width, height)
    }

    const renderFrame = () => {
      frameId = window.requestAnimationFrame(renderFrame)
      const time = clock.getElapsedTime()

      currentProgress += (progressRef.current - currentProgress) * 0.06
      noiseMaterial.uniforms.uTime.value = time
      fillMaterial.uniforms.uProgress.value = currentProgress / 100

      wireframeMaterial.opacity = 1 - currentProgress / 135
      const boxScale = 1 + currentProgress / 200
      wireframeBox.scale.set(boxScale, boxScale, boxScale)
      group.rotation.y = time * (0.16 + currentProgress * 0.012)
      group.rotation.x = Math.sin(time * 0.35) * 0.18
      group.rotation.z = Math.cos(time * 0.25) * 0.08

      renderer.render(scene, camera)
    }

    updateRendererSize()
    renderFrame()

    window.addEventListener('resize', updateRendererSize)

    return () => {
      window.cancelAnimationFrame(frameId)
      window.removeEventListener('resize', updateRendererSize)
      renderer.dispose()
      noiseGeometry.dispose()
      noiseMaterial.dispose()
      boxGeometry.dispose()
      boxEdges.dispose()
      wireframeMaterial.dispose()
      cylinderGeometry.dispose()
      cylinderEdges.dispose()
      fillMaterial.dispose()
      ;(cylinderWireframe.material as THREE.Material).dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className={styles.canvasContainer} aria-hidden="true" />
}

const AuthTransitionPage: React.FC = () => {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { isAuthenticated, isLoading } = useAuth()
  const { setupState } = useSetup()
  const [progress, setProgress] = useState(0)
  const [animationComplete, setAnimationComplete] = useState(false)

  const fallbackTarget = setupState?.setupMode ? '/setup' : '/dashboard'
  const target = sanitizeAuthTransitionTarget(searchParams.get('to'), fallbackTarget)
  const counterValue = Math.floor(progress)

  useEffect(() => {
    let frameId = 0
    const startTime = performance.now()

    const tick = (timestamp: number) => {
      const elapsed = timestamp - startTime
      const nextProgress = getTransitionProgress(elapsed)
      setProgress(nextProgress)

      if (elapsed >= AUTH_TRANSITION_MIN_DURATION_MS) {
        setAnimationComplete(true)
        setProgress(100)
        return
      }

      frameId = window.requestAnimationFrame(tick)
    }

    frameId = window.requestAnimationFrame(tick)

    return () => window.cancelAnimationFrame(frameId)
  }, [])

  useEffect(() => {
    if (animationComplete && isAuthenticated && !isLoading) {
      navigate(target, { replace: true })
    }
  }, [animationComplete, isAuthenticated, isLoading, navigate, target])

  if (!isAuthenticated && !isLoading) {
    return <Navigate to="/login" replace state={{ from: target }} />
  }

  return (
    <main className={styles.page}>
      <div className={styles.grid} aria-hidden="true">
        {Array.from({ length: 9 }, (_, index) => (
          <div key={index} className={styles.gridCell} />
        ))}
      </div>

      <TransitionScene progress={progress} />

      <div className={styles.overlay}>
        <section className={styles.topLeft}>
          <span className={styles.metaText}>SYS.ID: VSR-2025</span>
          <span className={styles.metaText}>INITIATING SEQUENCE</span>
        </section>

        <section className={styles.topRight} aria-hidden="true">
          <span className={styles.asterisk}>* * *</span>
        </section>

        <section className={styles.center}>
          <div className={styles.counterWrapper}>
            <span className={styles.counterNumber}>{counterValue}</span>
            <span className={styles.counterSymbol}>%</span>
          </div>
        </section>

        <section className={styles.bottomLeft}>
          <div className={styles.sequencePanel}>
            <span className={styles.sequenceHeader}>Signal lineage</span>
            <ol className={styles.sequenceList} aria-live="polite">
              {MILESTONES.map((milestone, index) => {
                const isVisible = progress >= milestone.revealAt
                const isActive =
                  isVisible &&
                  (index === MILESTONES.length - 1 || progress < MILESTONES[index + 1].revealAt)

                return (
                  <li
                    key={milestone.year}
                    className={`${styles.sequenceItem} ${isVisible ? styles.sequenceItemVisible : ''} ${isActive ? styles.sequenceItemActive : ''}`}
                  >
                    <span className={styles.sequenceYear}>{milestone.year}</span>
                    <span className={styles.sequenceCopy}>{milestone.text}</span>
                  </li>
                )
              })}
            </ol>
          </div>
        </section>

        <section className={styles.bottomRight}>
          <div className={styles.metaBlock}>
            <span className={styles.metaLabel}>VOL.</span>
            <span className={styles.metaValue}>04</span>
          </div>
        </section>

        <div className={styles.progressTrack} aria-hidden="true">
          <div className={styles.progressBar} style={{ width: `${progress}%` }} />
        </div>
      </div>
    </main>
  )
}

export default AuthTransitionPage
