import { useEffect, useRef } from 'react'
import { DASHBOARD_MOTION_COLORS } from '../components/dashboardMotionTheme'
import styles from './AuthTransitionPage.module.css'

interface AuthTransitionSceneProps {
  progress: number
  reducedMotion: boolean
}

const INBOUND_LEVELS = [-1.45, -0.9, -0.35, 0.35, 0.9, 1.45]

const AuthTransitionScene = ({ progress, reducedMotion }: AuthTransitionSceneProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const progressRef = useRef(progress)

  useEffect(() => {
    progressRef.current = progress
  }, [progress])

  useEffect(() => {
    let cancelled = false
    let disposeScene: (() => void) | undefined

    void import('three')
      .then((THREE) => {
        const container = containerRef.current
        if (!container || cancelled) return

        const scene = new THREE.Scene()
        const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100)
        camera.position.set(0, 0.2, 9.2)
        camera.lookAt(0, 0, 0)

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
        const geometries: Array<{ dispose: () => void }> = []
        const materials: Array<{ dispose: () => void }> = []
        let frameId: number | null = null
        let resizeObserver: ResizeObserver | undefined = undefined
        let disposed = false

        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75))
        renderer.setClearColor(0x000000, 0)
        renderer.domElement.style.display = 'block'

        disposeScene = () => {
          if (disposed) return
          disposed = true
          if (frameId !== null) window.cancelAnimationFrame(frameId)
          resizeObserver?.disconnect()
          geometries.forEach((geometry) => geometry.dispose())
          materials.forEach((material) => material.dispose())
          scene.clear()
          renderer.dispose()
          renderer.forceContextLoss()
          if (renderer.domElement.parentNode === container) {
            container.removeChild(renderer.domElement)
          }
        }

        container.appendChild(renderer.domElement)

        const loom = new THREE.Group()
        scene.add(loom)

        const ambientLight = new THREE.AmbientLight(DASHBOARD_MOTION_COLORS[0], 0.52)
        const keyLight = new THREE.PointLight(DASHBOARD_MOTION_COLORS[0], 11, 14)
        keyLight.position.set(1.4, 2.8, 4.8)
        const signalLight = new THREE.PointLight(DASHBOARD_MOTION_COLORS[1], 2.4, 4.5)
        const signalLightOffset = new THREE.Vector3(0, 0, 0.75)
        scene.add(ambientLight, keyLight, signalLight)

        const railMaterial = new THREE.LineBasicMaterial({
          color: DASHBOARD_MOTION_COLORS[2],
          transparent: true,
          opacity: 0.2,
        })
        const outputMaterial = new THREE.LineBasicMaterial({
          color: DASHBOARD_MOTION_COLORS[0],
          transparent: true,
          opacity: 0.26,
        })
        materials.push(railMaterial, outputMaterial)

        const addRail = (
          points: InstanceType<typeof THREE.Vector3>[],
          material: InstanceType<typeof THREE.LineBasicMaterial>,
        ) => {
          const geometry = new THREE.BufferGeometry().setFromPoints(points)
          geometries.push(geometry)
          loom.add(new THREE.Line(geometry, material))
        }

        INBOUND_LEVELS.forEach((level, index) => {
          const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(-4.8, level, -0.04 * index),
            new THREE.Vector3(-3.2, level * 0.96, 0),
            new THREE.Vector3(-1.55, level * 0.58, 0),
            new THREE.Vector3(-0.34, level * 0.11, 0),
          ])
          addRail(curve.getPoints(56), railMaterial)
        })
        ;[-0.52, 0.18, 0.88].forEach((level, index) => {
          const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(0.34, level * 0.12, 0),
            new THREE.Vector3(1.6, level * 0.48, 0),
            new THREE.Vector3(3.1, level * 0.85, 0),
            new THREE.Vector3(4.9, level, -0.03 * index),
          ])
          addRail(curve.getPoints(48), index === 1 ? outputMaterial : railMaterial)
        })

        const selectedCurve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(-4.8, -0.35, 0.08),
          new THREE.Vector3(-3.05, -0.34, 0.08),
          new THREE.Vector3(-1.45, -0.12, 0.08),
          new THREE.Vector3(-0.28, 0, 0.08),
          new THREE.Vector3(0.28, 0.03, 0.08),
          new THREE.Vector3(1.62, 0.12, 0.08),
          new THREE.Vector3(3.15, 0.17, 0.08),
          new THREE.Vector3(4.9, 0.18, 0.08),
        ])
        const selectedPoints = selectedCurve.getPoints(180)
        const selectedGeometry = new THREE.BufferGeometry().setFromPoints(selectedPoints)
        selectedGeometry.setDrawRange(0, 2)
        const selectedMaterial = new THREE.LineBasicMaterial({
          color: DASHBOARD_MOTION_COLORS[0],
          transparent: true,
          opacity: 0.94,
        })
        geometries.push(selectedGeometry)
        materials.push(selectedMaterial)
        loom.add(new THREE.Line(selectedGeometry, selectedMaterial))

        const gateVerticalGeometry = new THREE.BoxGeometry(0.042, 2.45, 0.11)
        const gateHorizontalGeometry = new THREE.BoxGeometry(0.52, 0.035, 0.11)
        const gateMaterial = new THREE.MeshStandardMaterial({
          color: DASHBOARD_MOTION_COLORS[2],
          emissive: DASHBOARD_MOTION_COLORS[0],
          emissiveIntensity: 0.08,
          metalness: 0.9,
          roughness: 0.24,
          transparent: true,
          opacity: 0.72,
        })
        geometries.push(gateVerticalGeometry, gateHorizontalGeometry)
        materials.push(gateMaterial)

        const gateLeft = new THREE.Mesh(gateVerticalGeometry, gateMaterial)
        const gateRight = new THREE.Mesh(gateVerticalGeometry, gateMaterial)
        const gateTop = new THREE.Mesh(gateHorizontalGeometry, gateMaterial)
        const gateBottom = new THREE.Mesh(gateHorizontalGeometry, gateMaterial)
        gateLeft.position.x = -0.26
        gateRight.position.x = 0.26
        gateTop.position.y = 1.2
        gateBottom.position.y = -1.2
        loom.add(gateLeft, gateRight, gateTop, gateBottom)

        const endpointGeometry = new THREE.BufferGeometry().setFromPoints([
          ...INBOUND_LEVELS.map((level) => new THREE.Vector3(-4.8, level, 0)),
          ...[-0.52, 0.18, 0.88].map((level) => new THREE.Vector3(4.9, level, 0)),
        ])
        const endpointMaterial = new THREE.PointsMaterial({
          color: DASHBOARD_MOTION_COLORS[0],
          size: 0.055,
          transparent: true,
          opacity: 0.42,
          sizeAttenuation: true,
        })
        geometries.push(endpointGeometry)
        materials.push(endpointMaterial)
        loom.add(new THREE.Points(endpointGeometry, endpointMaterial))

        const packetGeometry = new THREE.SphereGeometry(0.09, 18, 12)
        const packetMaterial = new THREE.MeshBasicMaterial({ color: DASHBOARD_MOTION_COLORS[1] })
        const packet = new THREE.Mesh(packetGeometry, packetMaterial)
        packet.visible = false
        geometries.push(packetGeometry)
        materials.push(packetMaterial)
        loom.add(packet)

        let currentProgress = 0
        const startTime = performance.now()

        const resize = () => {
          const width = container.clientWidth || window.innerWidth
          const height = container.clientHeight || window.innerHeight
          const compact = width < 700

          camera.aspect = width / Math.max(height, 1)
          camera.fov = compact ? 50 : 40
          camera.position.z = compact ? 10.4 : 9.2
          camera.updateProjectionMatrix()
          loom.position.set(compact ? 0 : 1.2, compact ? 0.65 : 0.1, 0)
          loom.scale.setScalar(compact ? 0.74 : 1)
          renderer.setSize(width, height, false)
        }

        const renderFrame = (staticFrame = false) => {
          const elapsed = staticFrame ? 0 : (performance.now() - startTime) / 1000
          currentProgress = staticFrame
            ? 100
            : currentProgress + (progressRef.current - currentProgress) * 0.09
          const normalizedProgress = Math.max(0, Math.min(currentProgress / 100, 1))
          const visiblePoints = Math.max(2, Math.floor(selectedPoints.length * normalizedProgress))
          selectedGeometry.setDrawRange(0, visiblePoints)

          const packetTravel = Math.max(0, Math.min((normalizedProgress - 0.04) / 1.08, 0.88))
          packet.position.copy(selectedCurve.getPointAt(packetTravel))
          packet.visible = normalizedProgress > 0.035
          packet.scale.setScalar(0.85 + Math.sin(elapsed * 3.2) * 0.08)
          signalLight.position.copy(packet.position).add(signalLightOffset)
          signalLight.intensity = packet.visible ? 2.1 + Math.sin(elapsed * 2.4) * 0.25 : 0

          const gateOpen = Math.sin(normalizedProgress * Math.PI) * 0.055
          gateLeft.position.x = -0.26 - gateOpen
          gateRight.position.x = 0.26 + gateOpen
          gateMaterial.emissiveIntensity = 0.08 + normalizedProgress * 0.12
          selectedMaterial.opacity = 0.56 + normalizedProgress * 0.38
          loom.rotation.y = staticFrame ? 0 : Math.sin(elapsed * 0.22) * 0.022
          loom.rotation.x = staticFrame ? 0 : Math.cos(elapsed * 0.18) * 0.012

          renderer.render(scene, camera)

          if (!staticFrame && !(progressRef.current >= 100 && currentProgress >= 99.6)) {
            frameId = window.requestAnimationFrame(() => renderFrame(false))
          }
        }

        resize()
        renderFrame(reducedMotion)

        resizeObserver = new ResizeObserver(() => {
          resize()
          if (reducedMotion) renderFrame(true)
        })
        resizeObserver.observe(container)

        if (cancelled) {
          disposeScene()
          disposeScene = undefined
        }
      })
      .catch(() => {
        disposeScene?.()
        disposeScene = undefined
      })

    return () => {
      cancelled = true
      disposeScene?.()
    }
  }, [reducedMotion])

  return (
    <div
      ref={containerRef}
      className={styles.canvasContainer}
      data-testid="auth-transition-scene"
      data-motion={reducedMotion ? 'static' : 'animated'}
      aria-hidden="true"
    />
  )
}

export default AuthTransitionScene
