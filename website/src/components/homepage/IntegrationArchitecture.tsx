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

const routerStages = [
  {
    id: 'signal',
    label: translate({ id: 'homepage.integration.pipeline.signal.label', message: 'Signal layer' }),
    detail: translate({ id: 'homepage.integration.pipeline.signal.detail', message: 'Detect' }),
  },
  {
    id: 'projection',
    label: translate({ id: 'homepage.integration.pipeline.projection.label', message: 'Projection layer' }),
    detail: translate({ id: 'homepage.integration.pipeline.projection.detail', message: 'Coordinate' }),
  },
  {
    id: 'decision',
    label: translate({ id: 'homepage.integration.pipeline.decision.label', message: 'Decision engine' }),
    detail: translate({ id: 'homepage.integration.pipeline.decision.detail', message: 'Match' }),
  },
  {
    id: 'algorithm',
    label: translate({ id: 'homepage.integration.pipeline.algorithm.label', message: 'Algorithms' }),
    detail: translate({ id: 'homepage.integration.pipeline.algorithm.detail', message: 'Optimize' }),
  },
  {
    id: 'plugins',
    label: translate({ id: 'homepage.integration.pipeline.plugins.label', message: 'Plugins' }),
    detail: translate({ id: 'homepage.integration.pipeline.plugins.detail', message: 'Enforce' }),
  },
] as const

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
  queries: Record<string, Point>
  stages: Record<string, Point>
  targets: Record<string, Point>
}

function toSvgPoint(point: Point): Point {
  return {
    x: (point.x / 100) * SVG_WIDTH,
    y: (point.y / 100) * SVG_HEIGHT,
  }
}

function buildRoutePath(points: Point[]): string {
  const svgPoints = points.map(toSvgPoint)
  const [start, ...rest] = svgPoints
  if (!start) {
    return ''
  }

  return rest.reduce((path, point, index) => {
    const previous = svgPoints[index]
    const distance = point.x - previous.x
    const controlOffset = Math.max(8, Math.abs(distance) * 0.42)

    return `${path} C ${previous.x + controlOffset},${previous.y} ${point.x - controlOffset},${point.y} ${point.x},${point.y}`
  }, `M ${start.x},${start.y}`)
}

function measureRouteLayout(
  diagram: HTMLElement,
  queryRows: HTMLElement[],
  stageRows: HTMLElement[],
  modelRows: HTMLElement[],
): RouteLayout | null {
  const diagramRect = diagram.getBoundingClientRect()
  if (diagramRect.width <= 0 || diagramRect.height <= 0) {
    return null
  }

  const toPercentX = (value: number) => ((value - diagramRect.left) / diagramRect.width) * 100
  const toPercentY = (value: number) => ((value - diagramRect.top) / diagramRect.height) * 100

  const queries: Record<string, Point> = {}
  incomingQueries.forEach((query, index) => {
    const row = queryRows[index]
    if (!row) {
      return
    }

    const rowRect = row.getBoundingClientRect()
    queries[query.id] = {
      x: toPercentX(rowRect.left + rowRect.width * 0.72),
      y: toPercentY(rowRect.top + rowRect.height / 2),
    }
  })

  const stages: Record<string, Point> = {}
  routerStages.forEach((stage, index) => {
    const row = stageRows[index]
    if (!row) {
      return
    }

    const rowRect = row.getBoundingClientRect()
    stages[stage.id] = {
      x: toPercentX(rowRect.left + rowRect.width / 2),
      y: toPercentY(rowRect.top + rowRect.height / 2),
    }
  })

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

  if (
    Object.keys(queries).length !== incomingQueries.length
    || Object.keys(stages).length !== routerStages.length
    || Object.keys(targets).length !== modelTargets.length
  ) {
    return null
  }

  return { queries, stages, targets }
}

function QueryColumn({
  title,
  activeQueryIndex,
  onRowRef,
}: {
  title: string
  activeQueryIndex: number
  onRowRef: (queryId: string, node: HTMLLIElement | null) => void
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
              ref={(node) => {
                onRowRef(query.id, node)
              }}
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

function RouterPipeline({
  logoSrc,
  activeQueryId,
  onStageRef,
}: {
  logoSrc: string
  activeQueryId: string
  onStageRef: (stageId: string, node: HTMLLIElement | null) => void
}): JSX.Element {
  return (
    <div className={styles.routerPipeline}>
      <header className={styles.pipelineHeader}>
        <img src={logoSrc} alt="vLLM Semantic Router" />
        <div>
          <span>
            <Translate id="homepage.integration.pipeline.eyebrow">Semantic routing pipeline</Translate>
          </span>
          <strong>
            <Translate id="homepage.integration.pipeline.title">Evidence to model selection</Translate>
          </strong>
        </div>
      </header>

      <ol className={styles.pipelineStages}>
        {routerStages.map((stage, index) => (
          <li
            key={`${activeQueryId}-${stage.id}`}
            ref={(node) => {
              onStageRef(stage.id, node)
            }}
            className={styles.pipelineStage}
            style={{
              '--stage-delay': `${0.28 + index * 0.43}s`,
            } as React.CSSProperties}
          >
            <span className={styles.pipelineStageIndex}>
              {String(index + 1).padStart(2, '0')}
            </span>
            <strong>{stage.label}</strong>
            <span className={styles.pipelineStageDetail}>{stage.detail}</span>
          </li>
        ))}
      </ol>

      <div className={styles.pipelineLayers}>
        <span>
          <Translate id="homepage.integration.layer.security">Security & policy</Translate>
        </span>
        <span>
          <Translate id="homepage.integration.layer.observability">Observability & replay</Translate>
        </span>
      </div>
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
  activeQuery,
  activeModel,
  layout,
}: {
  activeQuery: IncomingQuery
  activeModel: ModelTarget
  layout: RouteLayout | null
}): JSX.Element {
  const queryOrigin = layout?.queries[activeQuery.id]
  const target = layout?.targets[activeModel.id]
  const stagePoints = layout
    ? routerStages
        .map(stage => layout.stages[stage.id])
        .filter((point): point is Point => Boolean(point))
    : []
  const firstStage = stagePoints[0]
  const lastStage = stagePoints[stagePoints.length - 1]
  const activePath
    = queryOrigin && target && stagePoints.length === routerStages.length
      ? buildRoutePath([queryOrigin, ...stagePoints, target])
      : ''

  return (
    <div className={styles.flowTrack} aria-hidden="true">
      {layout && (
        <svg className={styles.flowLines} viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`} preserveAspectRatio="none">
          <defs>
            <linearGradient id="integrationFlowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#75c5ff" stopOpacity="0.55" />
              <stop offset="50%" stopColor="#30a2ff" stopOpacity="1" />
              <stop offset="100%" stopColor="#0876c9" stopOpacity="0.75" />
            </linearGradient>
            <filter id="integrationPacketGlow" x="-300%" y="-300%" width="700%" height="700%">
              <feGaussianBlur stdDeviation="3.2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {firstStage && incomingQueries.map((query) => {
            const origin = layout.queries[query.id]
            return origin
              ? (
                  <path
                    key={`inbound-${query.id}`}
                    className={styles.flowPathBase}
                    d={buildRoutePath([origin, firstStage])}
                  />
                )
              : null
          })}

          {stagePoints.length === routerStages.length && (
            <path
              className={styles.flowPathBase}
              d={buildRoutePath(stagePoints)}
            />
          )}

          {modelTargets.map((model) => {
            const routeTarget = layout.targets[model.id]
            if (!lastStage || !routeTarget) {
              return null
            }

            return (
              <path
                key={`egress-${model.id}`}
                className={styles.flowPathBase}
                d={buildRoutePath([lastStage, routeTarget])}
              />
            )
          })}

          {activePath && (
            <>
              <path
                key={`route-${activeQuery.id}-${activeModel.id}`}
                className={styles.flowPathActive}
                d={activePath}
                pathLength={1}
              />
              <circle
                key={`packet-${activeQuery.id}-${activeModel.id}`}
                className={styles.routePacket}
                r="4.5"
                filter="url(#integrationPacketGlow)"
              >
                <animateMotion
                  path={activePath}
                  dur="2.85s"
                  begin="0s"
                  fill="freeze"
                  calcMode="spline"
                  keyTimes="0;1"
                  keySplines="0.45 0 0.2 1"
                />
                <animate
                  attributeName="r"
                  values="3.5;5.5;4.5"
                  keyTimes="0;0.18;1"
                  dur="2.85s"
                  fill="freeze"
                />
              </circle>
            </>
          )}
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
  const queryRowRefs = useRef<Record<string, HTMLLIElement | null>>({})
  const stageRowRefs = useRef<Record<string, HTMLLIElement | null>>({})
  const modelRowRefs = useRef<Record<string, HTMLLIElement | null>>({})
  const activeQuery = incomingQueries[activeQueryIndex]
  const activeModel = getModelForQuery(activeQueryIndex)

  const handleQueryRowRef = useCallback((queryId: string, node: HTMLLIElement | null) => {
    queryRowRefs.current[queryId] = node
  }, [])

  const handleStageRowRef = useCallback((stageId: string, node: HTMLLIElement | null) => {
    stageRowRefs.current[stageId] = node
  }, [])

  const handleModelRowRef = useCallback((modelId: string, node: HTMLLIElement | null) => {
    modelRowRefs.current[modelId] = node
  }, [])

  const updateLayout = useCallback(() => {
    const diagram = diagramRef.current
    if (!diagram) {
      return
    }

    const queryRows = incomingQueries
      .map(query => queryRowRefs.current[query.id])
      .filter((row): row is HTMLLIElement => row !== null)
    const stageRows = routerStages
      .map(stage => stageRowRefs.current[stage.id])
      .filter((row): row is HTMLLIElement => row !== null)
    const modelRows = modelTargets
      .map(model => modelRowRefs.current[model.id])
      .filter((row): row is HTMLLIElement => row !== null)

    const nextLayout = measureRouteLayout(diagram, queryRows, stageRows, modelRows)
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
              <Translate id="homepage.integration.extproc.summary">Each request moves from signals through projections, decisions, algorithms, and plugins before reaching the best model pool. Clients keep the same OpenAI-compatible API.</Translate>
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
                activeQuery={activeQuery}
                activeModel={activeModel}
                layout={layout}
              />
              <QueryColumn
                title={translate({ id: 'homepage.integration.incoming', message: 'Incoming queries' })}
                activeQueryIndex={activeQueryIndex}
                onRowRef={handleQueryRowRef}
              />
              <RouterPipeline
                logoSrc={logoSrc}
                activeQueryId={activeQuery.id}
                onStageRef={handleStageRowRef}
              />
              <ModelColumn
                title={translate({ id: 'homepage.integration.models', message: 'Model pools' })}
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
