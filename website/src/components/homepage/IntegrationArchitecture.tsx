import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import Translate, { translate } from '@docusaurus/Translate'
import Link from '@docusaurus/Link'
import useBaseUrl from '@docusaurus/useBaseUrl'
import Claude from '@lobehub/icons/es/Claude/components/Mono'
import DeepSeek from '@lobehub/icons/es/DeepSeek/components/Mono'
import Gemini from '@lobehub/icons/es/Gemini/components/Mono'
import Mistral from '@lobehub/icons/es/Mistral/components/Mono'
import OpenAI from '@lobehub/icons/es/OpenAI/components/Mono'
import Zhipu from '@lobehub/icons/es/Zhipu/components/Mono'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import shared from './homepageShared.module.css'
import styles from './IntegrationArchitecture.module.css'

const extprocRouterModules = ['Signal layer', 'Decision engine', 'Plugins']
const DECISION_ENGINE_INDEX = 1

type ModelKind = 'closed' | 'open'

type ModelTarget = {
  id: string
  label: string
  provider: string
  kind: ModelKind
  Icon: React.ComponentType<{ size?: number }>
}

type IncomingQuery = {
  id: string
  label: string
}

const modelTargets: ModelTarget[] = [
  {
    id: 'anthropic',
    label: 'Claude',
    provider: 'Anthropic',
    kind: 'closed',
    Icon: Claude,
  },
  {
    id: 'openai',
    label: 'ChatGPT',
    provider: 'OpenAI',
    kind: 'closed',
    Icon: OpenAI,
  },
  {
    id: 'gemini',
    label: 'Gemini',
    provider: 'Google',
    kind: 'closed',
    Icon: Gemini,
  },
  {
    id: 'mistral',
    label: 'Mistral',
    provider: 'Mistral AI',
    kind: 'open',
    Icon: Mistral,
  },
  {
    id: 'deepseek',
    label: 'DeepSeek',
    provider: 'DeepSeek',
    kind: 'open',
    Icon: DeepSeek,
  },
  {
    id: 'glm',
    label: 'GLM',
    provider: 'Zhipu AI',
    kind: 'open',
    Icon: Zhipu,
  },
]

const incomingQueries: IncomingQuery[] = [
  {
    id: 'query-1',
    label: translate({ id: 'homepage.integration.query1', message: 'Query 1' }),
  },
  {
    id: 'query-2',
    label: translate({ id: 'homepage.integration.query2', message: 'Query 2' }),
  },
  {
    id: 'query-3',
    label: translate({ id: 'homepage.integration.query3', message: 'Query 3' }),
  },
  {
    id: 'query-4',
    label: translate({ id: 'homepage.integration.query4', message: 'Query 4' }),
  },
  {
    id: 'query-5',
    label: translate({ id: 'homepage.integration.query5', message: 'Query 5' }),
  },
  {
    id: 'query-6',
    label: translate({ id: 'homepage.integration.query6', message: 'Query 6' }),
  },
]

// Sequential queries, each routed to a different model in the fleet.
const ROUTE_TARGET_BY_QUERY = [3, 0, 4, 1, 5, 2]

function getModelForQuery(queryIndex: number): ModelTarget {
  const modelIndex = ROUTE_TARGET_BY_QUERY[queryIndex % ROUTE_TARGET_BY_QUERY.length]
  return modelTargets[modelIndex]
}

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
  modelTargets.forEach((model, index) => {
    const row = modelRows[index]
    if (!row) {
      return
    }

    const rowRect = row.getBoundingClientRect()
    targets[model.id] = {
      x: toPercentX(rowRect.left + rowRect.width * 0.5),
      y: toPercentY(rowRect.top + rowRect.height * 0.42),
    }
  })

  if (Object.keys(targets).length !== modelTargets.length) {
    return null
  }

  return { origin, targets }
}

function QueryColumn({
  title,
  activeQueryIndex,
}: {
  title: string
  activeQueryIndex: number
}): JSX.Element {
  return (
    <div className={styles.flowColumn}>
      <span className={styles.flowColumnTitle}>{title}</span>
      <ul className={styles.flowList}>
        {incomingQueries.map((query, index) => {
          const active = index === activeQueryIndex
          return (
            <li
              key={query.id}
              className={clsx(styles.flowNode, styles.query, {
                [styles.queryActive]: active,
              })}
              style={{ animationDelay: `${index * 0.15}s` }}
            >
              <span className={styles.queryLabel}>{query.label}</span>
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

function ModelTile({
  model,
  active,
  activeQueryLabel,
  onRowRef,
}: {
  model: ModelTarget
  active: boolean
  activeQueryLabel?: string
  onRowRef: (modelId: string, node: HTMLLIElement | null) => void
}): JSX.Element {
  const { Icon } = model
  const isOpen = model.kind === 'open'

  return (
    <li
      ref={(node) => {
        onRowRef(model.id, node)
      }}
      className={clsx(styles.modelTile, {
        [styles.modelTileOpen]: isOpen,
        [styles.modelTileActive]: active,
      })}
      aria-label={`${model.provider} ${model.label}`}
    >
      <span className={styles.modelTileArrow} aria-hidden="true" />
      <div className={styles.modelCube}>
        <div className={styles.modelCubeFace} aria-hidden="true" />
        <Icon size={isOpen ? 30 : 28} />
      </div>
      <span className={styles.modelTileName}>{model.label}</span>
      {active && activeQueryLabel && (
        <span className={styles.routeBadge}>
          {activeQueryLabel}
          {' '}
          →
        </span>
      )}
    </li>
  )
}

function ModelColumn({
  title,
  activeModelId,
  activeQueryLabel,
  onRowRef,
}: {
  title: string
  activeModelId: string
  activeQueryLabel: string
  onRowRef: (modelId: string, node: HTMLLIElement | null) => void
}): JSX.Element {
  const closedModels = modelTargets.filter(model => model.kind === 'closed')
  const openModels = modelTargets.filter(model => model.kind === 'open')

  return (
    <div className={styles.flowColumn}>
      <span className={styles.flowColumnTitle}>{title}</span>
      <div className={styles.modelShowcase}>
        <div className={styles.modelGroup}>
          <span className={styles.modelGroupLabel}>
            <Translate id="homepage.integration.closedModels">Closed models</Translate>
          </span>
          <ul className={styles.modelGrid}>
            {closedModels.map(model => (
              <ModelTile
                key={model.id}
                model={model}
                active={model.id === activeModelId}
                activeQueryLabel={activeQueryLabel}
                onRowRef={onRowRef}
              />
            ))}
          </ul>
        </div>

        <div className={clsx(styles.modelGroup, styles.modelGroupOpen)}>
          <span className={clsx(styles.modelGroupLabel, styles.modelGroupLabelOpen)}>
            <Translate id="homepage.integration.openModels">Open models</Translate>
          </span>
          <ul className={styles.modelGridOpen}>
            {openModels.map(model => (
              <ModelTile
                key={model.id}
                model={model}
                active={model.id === activeModelId}
                activeQueryLabel={activeQueryLabel}
                onRowRef={onRowRef}
              />
            ))}
          </ul>
        </div>

        <p className={styles.heterogeneousCaption}>
          <Translate id="homepage.integration.heterogeneousModels">Heterogeneous models</Translate>
        </p>
      </div>
    </div>
  )
}

function RoutingAnimation({
  activeModel,
  activeQueryLabel,
  layout,
}: {
  activeModel: ModelTarget
  activeQueryLabel: string
  layout: RouteLayout | null
}): JSX.Element {
  const packetRef = useRef<HTMLSpanElement | null>(null)
  const target = layout?.targets[activeModel.id]
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
  }, [activeModel.id, origin, target, mid])

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
          key={`${activeModel.id}-${activeQueryLabel}`}
          ref={packetRef}
          className={styles.queryPacket}
          style={{
            left: `${origin.x}%`,
            top: `${origin.y}%`,
          }}
        >
          <span className={styles.packetLabel}>{activeQueryLabel}</span>
        </span>
      )}
      {layout && (
        <svg className={styles.flowLines} viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`} preserveAspectRatio="none">
          <defs>
            <linearGradient id="integrationFlowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#75c5ff" stopOpacity="0.55" />
              <stop offset="50%" stopColor="#30a2ff" stopOpacity="1" />
              <stop offset="100%" stopColor="#0876c9" stopOpacity="0.75" />
            </linearGradient>
          </defs>
          {modelTargets.map((model) => {
            const routeTarget = layout.targets[model.id]
            if (!routeTarget) {
              return null
            }

            return (
              <path
                key={model.id}
                className={clsx(styles.flowPathEgress, {
                  [styles.flowPathEgressActive]: activeModel.id === model.id,
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
  const [activeQueryIndex, setActiveQueryIndex] = useState(0)
  const [layout, setLayout] = useState<RouteLayout | null>(null)
  const logoSrc = useBaseUrl('/img/vllm-sr-logo.white.png')
  const diagramRef = useRef<HTMLDivElement | null>(null)
  const decisionEngineRef = useRef<HTMLLIElement | null>(null)
  const modelRowRefs = useRef<Record<string, HTMLLIElement | null>>({})
  const activeQuery = incomingQueries[activeQueryIndex]
  const activeModel = getModelForQuery(activeQueryIndex)

  const handleModelRowRef = useCallback((modelId: string, node: HTMLLIElement | null) => {
    modelRowRefs.current[modelId] = node
  }, [])

  const updateLayout = useCallback(() => {
    const diagram = diagramRef.current
    const decisionEngine = decisionEngineRef.current
    if (!diagram || !decisionEngine) {
      return
    }

    const modelRows = modelTargets
      .map(model => modelRowRefs.current[model.id])
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
  }, [updateLayout, activeQueryIndex])

  useEffect(() => {
    const retryDelays = [80, 200, 500, 1000]
    const timeoutIds = retryDelays.map(delay =>
      window.setTimeout(() => {
        updateLayout()
      }, delay),
    )

    return () => {
      timeoutIds.forEach((timeoutId) => {
        window.clearTimeout(timeoutId)
      })
    }
  }, [updateLayout, activeQueryIndex])

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setActiveQueryIndex(current => (current + 1) % incomingQueries.length)
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
              <span className={styles.routeStatusValue} key={`${activeQuery.id}-${activeModel.id}`}>
                {activeQuery.label}
                {' '}
                →
                {' '}
                {activeModel.label}
              </span>
            </div>

            <div className={styles.diagramShell} ref={diagramRef}>
              <RoutingAnimation
                activeModel={activeModel}
                activeQueryLabel={activeQuery.label}
                layout={layout}
              />
              <QueryColumn
                title={translate({ id: 'homepage.integration.incoming', message: 'Incoming queries' })}
                activeQueryIndex={activeQueryIndex}
              />
              <div className={`${styles.routerHub} ${styles.routerHubActive}`}>
                <div className={styles.hubBrand}>
                  <img src={logoSrc} alt="vLLM Semantic Router" />
                </div>
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
                activeModelId={activeModel.id}
                activeQueryLabel={activeQuery.label}
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
