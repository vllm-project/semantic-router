import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import Translate, { translate } from '@docusaurus/Translate'
import Link from '@docusaurus/Link'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import shared from './homepageShared.module.css'
import styles from './IntegrationArchitecture.module.css'

const extprocRouterModules = ['Signal layer', 'Decision engine', 'Plugins']
const DECISION_ENGINE_INDEX = 1

type ModelRoute = {
  id: string
  label: string
  queryLabel: string
}

const modelRoutes: ModelRoute[] = [
  {
    id: 'claude',
    label: 'Claude',
    queryLabel: translate({ id: 'homepage.integration.query1', message: 'Query 1' }),
  },
  {
    id: 'chatgpt',
    label: 'ChatGPT',
    queryLabel: translate({ id: 'homepage.integration.query2', message: 'Query 2' }),
  },
  {
    id: 'llama',
    label: 'Llama',
    queryLabel: translate({ id: 'homepage.integration.query3', message: 'Query 3' }),
  },
]

const ROUTE_CYCLE_MS = 3400
const SVG_WIDTH = 800
const SVG_HEIGHT = 200

type Point = {
  x: number
  y: number
}

type RouteLayout = {
  origin: Point
  targets: Record<string, Point>
}

function toSvgPoint(point: Point): Point {
  return {
    x: (point.x / 100) * SVG_WIDTH,
    y: (point.y / 100) * SVG_HEIGHT,
  }
}

function buildEgressPath(origin: Point, target: Point): string {
  const start = toSvgPoint(origin)
  const end = toSvgPoint(target)
  const control1X = start.x + (end.x - start.x) * 0.42
  const control1Y = start.y
  const control2X = end.x - (end.x - start.x) * 0.22
  const control2Y = end.y

  return `M ${start.x},${start.y} C ${control1X},${control1Y} ${control2X},${control2Y} ${end.x},${end.y}`
}

function measureRouteLayout(
  diagram: HTMLElement,
  decisionEngine: HTMLElement,
  modelRows: HTMLElement[],
): RouteLayout | null {
  const diagramRect = diagram.getBoundingClientRect()
  if (diagramRect.width <= 0 || diagramRect.height <= 0) {
    return null
  }

  const toPercentX = (value: number) => ((value - diagramRect.left) / diagramRect.width) * 100
  const toPercentY = (value: number) => ((value - diagramRect.top) / diagramRect.height) * 100

  const decisionRect = decisionEngine.getBoundingClientRect()
  const origin = {
    x: toPercentX(decisionRect.left + decisionRect.width / 2),
    y: toPercentY(decisionRect.top + decisionRect.height / 2),
  }

  const targets: Record<string, Point> = {}
  modelRoutes.forEach((route, index) => {
    const row = modelRows[index]
    if (!row) {
      return
    }

    const rowRect = row.getBoundingClientRect()
    targets[route.id] = {
      x: toPercentX(rowRect.left + rowRect.width * 0.42),
      y: toPercentY(rowRect.top + rowRect.height / 2),
    }
  })

  if (Object.keys(targets).length !== modelRoutes.length) {
    return null
  }

  return { origin, targets }
}

function QueryColumn({
  title,
  activeRouteId,
}: {
  title: string
  activeRouteId: string
}): JSX.Element {
  return (
    <div className={styles.flowColumn}>
      <span className={styles.flowColumnTitle}>{title}</span>
      <ul className={styles.flowList}>
        {modelRoutes.map((route, index) => {
          const active = route.id === activeRouteId
          return (
            <li
              key={route.id}
              className={clsx(styles.flowNode, styles.query, {
                [styles.queryActive]: active,
              })}
              style={{ animationDelay: `${index * 0.15}s` }}
            >
              <span className={styles.queryLabel}>{route.queryLabel}</span>
              {active && (
                <span className={styles.querySending}>
                  <Translate id="homepage.integration.sending">Sending</Translate>
                </span>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function ModelColumn({
  title,
  activeRouteId,
  onRowRef,
}: {
  title: string
  activeRouteId: string
  onRowRef: (routeId: string, node: HTMLLIElement | null) => void
}): JSX.Element {
  return (
    <div className={styles.flowColumn}>
      <span className={styles.flowColumnTitle}>{title}</span>
      <ul className={styles.flowList}>
        {modelRoutes.map((route) => {
          const active = route.id === activeRouteId
          return (
            <li
              key={route.id}
              ref={(node) => {
                onRowRef(route.id, node)
              }}
              className={clsx(styles.flowNode, styles.model, {
                [styles.modelActive]: active,
              })}
            >
              <span className={styles.modelLabel}>{route.label}</span>
              {active && (
                <span className={styles.routeBadge}>
                  {route.queryLabel}
                  {' '}
                  →
                </span>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function RoutingAnimation({
  activeRoute,
  layout,
}: {
  activeRoute: ModelRoute
  layout: RouteLayout | null
}): JSX.Element {
  const packetRef = useRef<HTMLSpanElement | null>(null)
  const target = layout?.targets[activeRoute.id]
  const origin = layout?.origin
  const mid = origin && target
    ? {
        x: origin.x + (target.x - origin.x) * 0.3,
        y: origin.y + (target.y - origin.y) * 0.4,
      }
    : null

  useLayoutEffect(() => {
    const packet = packetRef.current
    if (!packet || !origin || !target || !mid) {
      return undefined
    }

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReducedMotion) {
      packet.style.left = `${target.x}%`
      packet.style.top = `${target.y}%`
      packet.style.opacity = '1'
      packet.style.transform = 'translate(-50%, -50%)'
      return undefined
    }

    const keyframes: Keyframe[] = [
      {
        left: `${origin.x}%`,
        top: `${origin.y}%`,
        opacity: 0,
        transform: 'translate(-50%, -50%) scale(0.6)',
      },
      {
        opacity: 1,
        transform: 'translate(-50%, -50%) scale(1)',
        offset: 0.1,
      },
      {
        left: `${origin.x}%`,
        top: `${origin.y}%`,
        opacity: 1,
        transform: 'translate(-50%, -50%) scale(1)',
        offset: 0.3,
      },
      {
        left: `${mid.x}%`,
        top: `${mid.y}%`,
        opacity: 1,
        transform: 'translate(-50%, -50%) scale(1)',
        offset: 0.55,
      },
      {
        left: `${target.x}%`,
        top: `${target.y}%`,
        opacity: 1,
        transform: 'translate(-50%, -50%) scale(1)',
        offset: 0.82,
      },
      {
        left: `${target.x}%`,
        top: `${target.y}%`,
        opacity: 0,
        transform: 'translate(-50%, -50%) scale(0.7)',
        offset: 1,
      },
    ]

    const animation = packet.animate(keyframes, {
      duration: 3200,
      easing: 'ease-in-out',
      fill: 'forwards',
    })

    return () => {
      animation.cancel()
    }
  }, [activeRoute.id, origin, target, mid])

  return (
    <div className={styles.flowTrack} aria-hidden="true">
      {origin && (
        <span
          className={styles.routeOrigin}
          style={{
            left: `${origin.x}%`,
            top: `${origin.y}%`,
          }}
        />
      )}
      {origin && target && (
        <span
          key={activeRoute.id}
          ref={packetRef}
          className={styles.queryPacket}
          style={{
            left: `${origin.x}%`,
            top: `${origin.y}%`,
          }}
        >
          <span className={styles.packetLabel}>{activeRoute.queryLabel}</span>
        </span>
      )}
      {layout && (
        <svg className={styles.flowLines} viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`} preserveAspectRatio="none">
          <defs>
            <linearGradient id="integrationFlowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.45" />
              <stop offset="50%" stopColor="#a78bfa" stopOpacity="1" />
              <stop offset="100%" stopColor="#c4b5fd" stopOpacity="0.65" />
            </linearGradient>
          </defs>
          {modelRoutes.map((route) => {
            const routeTarget = layout.targets[route.id]
            if (!routeTarget) {
              return null
            }

            return (
              <path
                key={route.id}
                className={clsx(styles.flowPathEgress, {
                  [styles.flowPathEgressActive]: activeRoute.id === route.id,
                })}
                d={buildEgressPath(layout.origin, routeTarget)}
              />
            )
          })}
        </svg>
      )}
    </div>
  )
}

export default function IntegrationArchitecture(): JSX.Element {
  const [routeIndex, setRouteIndex] = useState(0)
  const [layout, setLayout] = useState<RouteLayout | null>(null)
  const diagramRef = useRef<HTMLDivElement | null>(null)
  const decisionEngineRef = useRef<HTMLLIElement | null>(null)
  const modelRowRefs = useRef<Record<string, HTMLLIElement | null>>({})
  const activeRoute = modelRoutes[routeIndex]

  const handleModelRowRef = useCallback((routeId: string, node: HTMLLIElement | null) => {
    modelRowRefs.current[routeId] = node
  }, [])

  const updateLayout = useCallback(() => {
    const diagram = diagramRef.current
    const decisionEngine = decisionEngineRef.current
    if (!diagram || !decisionEngine) {
      return
    }

    const modelRows = modelRoutes
      .map(route => modelRowRefs.current[route.id])
      .filter((row): row is HTMLLIElement => row !== null)

    const nextLayout = measureRouteLayout(diagram, decisionEngine, modelRows)
    if (nextLayout) {
      setLayout(nextLayout)
    }
  }, [])

  useLayoutEffect(() => {
    updateLayout()

    const diagram = diagramRef.current
    if (!diagram) {
      return undefined
    }

    const frameId = window.requestAnimationFrame(() => {
      updateLayout()
    })

    if (typeof ResizeObserver === 'undefined') {
      return () => {
        window.cancelAnimationFrame(frameId)
      }
    }

    const resizeObserver = new ResizeObserver(() => {
      updateLayout()
    })
    resizeObserver.observe(diagram)

    return () => {
      window.cancelAnimationFrame(frameId)
      resizeObserver.disconnect()
    }
  }, [updateLayout])

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setRouteIndex(current => (current + 1) % modelRoutes.length)
    }, ROUTE_CYCLE_MS)

    return () => {
      window.clearInterval(intervalId)
    }
  }, [])

  return (
    <section className={shared.bandSection} aria-labelledby="integration-architecture-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <ScrollReveal>
          <header className={shared.sectionHeader}>
            <span className={shared.eyebrow}>
              <Translate id="homepage.integration.eyebrow">How it integrates</Translate>
            </span>
            <h2 id="integration-architecture-title" className={shared.sectionTitle}>
              <Translate id="homepage.integration.title">Route queries to the right model</Translate>
            </h2>
            <p className={shared.sectionSubtitle}>
              <Translate id="homepage.integration.extproc.summary">The decision engine classifies each request and picks the best model in your fleet. Clients keep the same OpenAI-compatible API.</Translate>
            </p>
          </header>
        </ScrollReveal>

        <ScrollReveal delay={80}>
          <div className={`${shared.darkCard} ${styles.shell}`}>
            <div className={styles.routeStatus} aria-live="polite">
              <span className={styles.routeStatusLabel}>
                <Translate id="homepage.integration.routing">Routing</Translate>
              </span>
              <span className={styles.routeStatusValue} key={activeRoute.id}>
                {activeRoute.queryLabel}
                {' '}
                →
                {' '}
                {activeRoute.label}
              </span>
            </div>

            <div className={styles.diagramShell} ref={diagramRef}>
              <RoutingAnimation activeRoute={activeRoute} layout={layout} />
              <QueryColumn
                title={translate({ id: 'homepage.integration.incoming', message: 'Incoming queries' })}
                activeRouteId={activeRoute.id}
              />
              <div className={`${styles.routerHub} ${styles.routerHubActive}`}>
                <span className={styles.hubBadge}>
                  <Translate id="homepage.integration.hub">vLLM Semantic Router</Translate>
                </span>
                <ul className={styles.hubModules}>
                  {extprocRouterModules.map((module, index) => (
                    <li
                      key={module}
                      ref={index === DECISION_ENGINE_INDEX ? decisionEngineRef : undefined}
                      style={{ animationDelay: `${index * 0.2}s` }}
                    >
                      {module}
                    </li>
                  ))}
                </ul>
                <div className={styles.hubLayers}>
                  <span>
                    <Translate id="homepage.integration.layer.security">Security & policy</Translate>
                  </span>
                  <span>
                    <Translate id="homepage.integration.layer.observability">Observability & replay</Translate>
                  </span>
                </div>
                <span className={styles.hubPulse} aria-hidden="true" />
              </div>
              <ModelColumn
                title={translate({ id: 'homepage.integration.models', message: 'Models' })}
                activeRouteId={activeRoute.id}
                onRowRef={handleModelRowRef}
              />
            </div>

            <div className={styles.footer}>
              <span className={styles.compatPill}>
                <Translate id="homepage.integration.extproc.compat">
                  Backward compatible — no client changes required
                </Translate>
              </span>
              <Link className={styles.docsLink} to="/docs/installation">
                <Translate id="homepage.integration.viewDocs">View integration guide</Translate>
                {' '}
                →
              </Link>
            </div>
          </div>
        </ScrollReveal>
      </div>
    </section>
  )
}
