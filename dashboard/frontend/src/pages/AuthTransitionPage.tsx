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
  key: string
  label: string
  detail: string
  revealAt: number
}

const PROGRESS_SEGMENTS = [
  { duration: 520, from: 0, to: 24 },
  { duration: 620, from: 24, to: 54 },
  { duration: 680, from: 54, to: 84 },
  { duration: 480, from: 84, to: 100 },
]

const MILESTONES: Milestone[] = [
  {
    key: 'session',
    label: 'Session verified',
    detail: 'Credentials accepted',
    revealAt: 8,
  },
  {
    key: 'workspace',
    label: 'Workspace synced',
    detail: 'Setup state loaded',
    revealAt: 34,
  },
  {
    key: 'policy',
    label: 'Policy hydrated',
    detail: 'Routes and signals prepared',
    revealAt: 62,
  },
  {
    key: 'dashboard',
    label: 'Dashboard ready',
    detail: 'Opening console',
    revealAt: 86,
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

function getActiveMilestoneIndex(progress: number): number {
  return MILESTONES.reduce((activeIndex, milestone, index) => {
    return progress >= milestone.revealAt ? index : activeIndex
  }, 0)
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
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100)
    camera.position.set(0, 0.9, 8.8)
    camera.lookAt(0, 0, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x000000, 0)
    renderer.domElement.style.display = 'block'
    container.appendChild(renderer.domElement)

    const group = new THREE.Group()
    scene.add(group)

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.48)
    scene.add(ambientLight)

    const keyLight = new THREE.PointLight(0x8fd400, 18, 12)
    keyLight.position.set(-2.6, 2.4, 4)
    scene.add(keyLight)

    const cyanLight = new THREE.PointLight(0x00d4ff, 10, 12)
    cyanLight.position.set(2.8, -1.4, 3)
    scene.add(cyanLight)

    const nodePositions = [
      new THREE.Vector3(-3.2, -0.65, 0),
      new THREE.Vector3(-1.05, 0.72, 0),
      new THREE.Vector3(1.14, -0.18, 0),
      new THREE.Vector3(3.08, 0.58, 0),
    ]

    const curve = new THREE.CatmullRomCurve3(nodePositions)
    const curvePoints = curve.getPoints(96)
    const routeGeometry = new THREE.BufferGeometry().setFromPoints(curvePoints)
    const routeMaterial = new THREE.LineBasicMaterial({
      color: 0x8fd400,
      transparent: true,
      opacity: 0.28,
    })
    const routeLine = new THREE.Line(routeGeometry, routeMaterial)
    group.add(routeLine)

    const completedRouteGeometry = new THREE.BufferGeometry().setFromPoints([nodePositions[0], nodePositions[0]])
    const completedRouteMaterial = new THREE.LineBasicMaterial({
      color: 0xb7ff56,
      transparent: true,
      opacity: 0.86,
    })
    const completedRouteLine = new THREE.Line(completedRouteGeometry, completedRouteMaterial)
    group.add(completedRouteLine)

    const nodeGeometry = new THREE.SphereGeometry(0.17, 32, 16)
    const nodeMaterials = nodePositions.map(() =>
      new THREE.MeshStandardMaterial({
        color: 0x76b900,
        emissive: 0x76b900,
        emissiveIntensity: 0.15,
        metalness: 0.18,
        roughness: 0.48,
        transparent: true,
        opacity: 0.5,
      }),
    )
    const nodeMeshes = nodePositions.map((position, index) => {
      const mesh = new THREE.Mesh(nodeGeometry, nodeMaterials[index])
      mesh.position.copy(position)
      group.add(mesh)
      return mesh
    })

    const nodeRingGeometry = new THREE.RingGeometry(0.28, 0.31, 40)
    const nodeRingMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.2,
      side: THREE.DoubleSide,
    })
    const nodeRings = nodePositions.map((position) => {
      const ring = new THREE.Mesh(nodeRingGeometry, nodeRingMaterial.clone())
      ring.position.copy(position)
      ring.lookAt(camera.position)
      group.add(ring)
      return ring
    })

    const particleCount = 22
    const particleGeometry = new THREE.SphereGeometry(0.045, 16, 8)
    const particleMaterial = new THREE.MeshBasicMaterial({
      color: 0xd9ff8a,
      transparent: true,
      opacity: 0.78,
    })
    const particles = Array.from({ length: particleCount }, (_, index) => {
      const particle = new THREE.Mesh(particleGeometry, particleMaterial.clone())
      particle.userData.offset = index / particleCount
      group.add(particle)
      return particle
    })

    const arcGeometry = new THREE.TorusGeometry(1.65, 0.01, 8, 128, Math.PI * 1.42)
    const arcMaterial = new THREE.MeshBasicMaterial({
      color: 0x76b900,
      transparent: true,
      opacity: 0.42,
    })
    const progressArc = new THREE.Mesh(arcGeometry, arcMaterial)
    progressArc.position.set(0, 0, -0.28)
    progressArc.rotation.set(0, 0, -Math.PI * 0.18)
    group.add(progressArc)

    const haloGeometry = new THREE.RingGeometry(1.98, 2, 128)
    const haloMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.08,
      side: THREE.DoubleSide,
    })
    const halo = new THREE.Mesh(haloGeometry, haloMaterial)
    halo.position.set(0, 0, -0.34)
    halo.lookAt(camera.position)
    group.add(halo)

    const startTime = performance.now()
    let frameId = 0
    let currentProgress = 0

    const updateRendererSize = () => {
      const width = container.clientWidth || window.innerWidth
      const height = container.clientHeight || window.innerHeight

      camera.aspect = width / Math.max(height, 1)
      camera.fov = width < 640 ? 52 : 42
      camera.position.z = width < 640 ? 9.8 : 8.8
      camera.updateProjectionMatrix()

      renderer.setSize(width, height, false)
    }

    const renderFrame = () => {
      frameId = window.requestAnimationFrame(renderFrame)
      const time = (performance.now() - startTime) / 1000

      currentProgress += (progressRef.current - currentProgress) * 0.08
      const normalizedProgress = Math.max(0, Math.min(currentProgress / 100, 1))
      const visiblePointCount = Math.max(2, Math.floor(curvePoints.length * normalizedProgress))
      completedRouteGeometry.setFromPoints(curvePoints.slice(0, visiblePointCount))

      keyLight.intensity = 15 + Math.sin(time * 1.4) * 2
      cyanLight.intensity = 8 + Math.cos(time * 1.1) * 1.5
      group.rotation.y = Math.sin(time * 0.28) * 0.12
      group.rotation.x = Math.cos(time * 0.24) * 0.06

      progressArc.scale.setScalar(0.92 + normalizedProgress * 0.24)
      arcMaterial.opacity = 0.28 + normalizedProgress * 0.34
      halo.rotation.z = time * 0.08
      haloMaterial.opacity = 0.06 + Math.sin(time * 1.8) * 0.015

      nodeMeshes.forEach((mesh, index) => {
        const threshold = MILESTONES[index]?.revealAt ?? 100
        const activeAmount = Math.max(0, Math.min((currentProgress - threshold + 16) / 22, 1))
        const pulse = Math.sin(time * 2.2 + index * 0.8) * 0.035
        const scale = 0.88 + activeAmount * 0.56 + pulse
        mesh.scale.setScalar(scale)
        nodeMaterials[index].opacity = 0.38 + activeAmount * 0.62
        nodeMaterials[index].emissiveIntensity = 0.12 + activeAmount * 0.82
      })

      nodeRings.forEach((ring, index) => {
        const threshold = MILESTONES[index]?.revealAt ?? 100
        const activeAmount = Math.max(0, Math.min((currentProgress - threshold + 12) / 20, 1))
        ring.lookAt(camera.position)
        ring.rotation.z += 0.01 + activeAmount * 0.012
        ring.scale.setScalar(0.8 + activeAmount * 0.8 + Math.sin(time * 1.7 + index) * 0.04)
        ;(ring.material as THREE.MeshBasicMaterial).opacity = 0.1 + activeAmount * 0.38
      })

      particles.forEach((particle, index) => {
        const baseOffset = particle.userData.offset as number
        const travel = (baseOffset + time * 0.13) % 1
        const gatedTravel = Math.min(travel, normalizedProgress)
        const point = curve.getPointAt(gatedTravel)
        const wobble = Math.sin(time * 2.2 + index) * 0.035
        particle.position.set(point.x, point.y + wobble, point.z + Math.cos(time + index) * 0.025)
        const particleActive = travel <= normalizedProgress || normalizedProgress > 0.96
        ;(particle.material as THREE.MeshBasicMaterial).opacity = particleActive ? 0.28 + normalizedProgress * 0.62 : 0
        particle.scale.setScalar(0.7 + Math.sin(time * 3 + index) * 0.16)
      })

      renderer.render(scene, camera)
    }

    updateRendererSize()
    renderFrame()

    window.addEventListener('resize', updateRendererSize)

    return () => {
      window.cancelAnimationFrame(frameId)
      window.removeEventListener('resize', updateRendererSize)
      renderer.dispose()
      routeGeometry.dispose()
      routeMaterial.dispose()
      completedRouteGeometry.dispose()
      completedRouteMaterial.dispose()
      nodeGeometry.dispose()
      nodeMaterials.forEach((material) => material.dispose())
      nodeRingGeometry.dispose()
      nodeRings.forEach((ring) => (ring.material as THREE.Material).dispose())
      particleGeometry.dispose()
      particles.forEach((particle) => (particle.material as THREE.Material).dispose())
      arcGeometry.dispose()
      arcMaterial.dispose()
      haloGeometry.dispose()
      haloMaterial.dispose()
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
  const activeMilestoneIndex = getActiveMilestoneIndex(progress)
  const activeMilestone = MILESTONES[activeMilestoneIndex]

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
    <main className={styles.page} aria-busy={!animationComplete} aria-live="polite" role="status">
      <div className={styles.grid} aria-hidden="true">
        {Array.from({ length: 9 }, (_, index) => (
          <div key={index} className={styles.gridCell} />
        ))}
      </div>

      <TransitionScene progress={progress} />

      <div className={styles.overlay}>
        <section className={styles.topLeft}>
          <span className={styles.metaText}>VSR Dashboard</span>
          <span className={styles.metaText}>{activeMilestone.label}</span>
        </section>

        <section className={styles.topRight} aria-hidden="true">
          <span className={styles.statusBeacon} />
          <span className={styles.metaText}>Live route handoff</span>
        </section>

        <section className={styles.center}>
          <div className={styles.centerCopy}>
            <span className={styles.centerEyebrow}>Authenticated</span>
            <h1 className={styles.centerTitle}>Opening dashboard</h1>
            <p className={styles.centerText}>{activeMilestone.detail}</p>
          </div>
        </section>

        <section className={styles.bottomLeft}>
          <div className={styles.sequencePanel}>
            <span className={styles.sequenceHeader}>Loading sequence</span>
            <ol className={styles.sequenceList} aria-live="polite">
              {MILESTONES.map((milestone, index) => {
                const isVisible = progress >= milestone.revealAt
                const isActive = index === activeMilestoneIndex

                return (
                  <li
                    key={milestone.key}
                    className={`${styles.sequenceItem} ${isVisible ? styles.sequenceItemVisible : ''} ${isActive ? styles.sequenceItemActive : ''}`}
                  >
                    <span className={styles.sequenceYear}>{String(index + 1).padStart(2, '0')}</span>
                    <span>
                      <span className={styles.sequenceCopy}>{milestone.label}</span>
                      <span className={styles.sequenceDetail}>{milestone.detail}</span>
                    </span>
                  </li>
                )
              })}
            </ol>
          </div>
        </section>

        <section className={styles.bottomRight}>
          <div className={styles.metaBlock}>
            <span className={styles.metaLabel}>Progress</span>
            <span className={styles.metaValue}>{Math.round(progress)}%</span>
          </div>
        </section>

        <div
          className={styles.progressTrack}
          role="progressbar"
          aria-label="Opening dashboard"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(progress)}
        >
          <div className={styles.progressBar} style={{ transform: `scaleX(${progress / 100})` }} />
        </div>
      </div>
    </main>
  )
}

export default AuthTransitionPage
