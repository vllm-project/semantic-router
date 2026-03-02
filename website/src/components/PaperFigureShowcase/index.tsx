import React, { useEffect, useMemo, useRef, useState } from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import { motion, useInView } from 'motion/react'
import styles from './index.module.css'

type FigureKey = 'figure1' | 'figure2' | 'figure3' | 'figure4' | 'figure14' | 'figure9'

type FigureMeta = {
  key: FigureKey
  paperLabel: string
  index: string
  title: React.ReactNode
  summary: React.ReactNode
}

const FIGURES: FigureMeta[] = [
  {
    key: 'figure1',
    paperLabel: 'fig:shannon_mapping',
    index: '01',
    title: <Translate id="homepage.paperFigures.figure1.title">Shannon Mapping</Translate>,
    summary: (
      <Translate id="homepage.paperFigures.figure1.summary">
        Structural mapping from communication theory to the routing pipeline.
      </Translate>
    ),
  },
  {
    key: 'figure2',
    paperLabel: 'fig:entropy_collapse',
    index: '02',
    title: <Translate id="homepage.paperFigures.figure2.title">Entropy Collapse</Translate>,
    summary: (
      <Translate id="homepage.paperFigures.figure2.summary">
        Each additional signal reduces uncertainty until deterministic model selection.
      </Translate>
    ),
  },
  {
    key: 'figure3',
    paperLabel: 'fig:architecture',
    index: '03',
    title: <Translate id="homepage.paperFigures.figure3.title">Three-Layer Architecture</Translate>,
    summary: (
      <Translate id="homepage.paperFigures.figure3.summary">
        Signal extraction, decision engine, and plugin projection with closed-loop feedback.
      </Translate>
    ),
  },
  {
    key: 'figure4',
    paperLabel: 'fig:signal_taxonomy',
    index: '04',
    title: <Translate id="homepage.paperFigures.figure4.title">Signal Taxonomy</Translate>,
    summary: (
      <Translate id="homepage.paperFigures.figure4.summary">
        All 13 signals are grouped into heuristic and learned paths, then merged into S(r).
      </Translate>
    ),
  },
  {
    key: 'figure14',
    paperLabel: 'fig:agent_synthesis',
    index: '05',
    title: <Translate id="homepage.paperFigures.figure14.title">Agent Policy Synthesis</Translate>,
    summary: (
      <Translate id="homepage.paperFigures.figure14.summary">
        Natural-language routing intent is synthesized into DSL policy, executed, and refined by feedback.
      </Translate>
    ),
  },
  {
    key: 'figure9',
    paperLabel: 'fig:entropy_folding_layers',
    index: '09',
    title: <Translate id="homepage.paperFigures.figure9.title">Layered Entropy Folding</Translate>,
    summary: (
      <Translate id="homepage.paperFigures.figure9.summary">
        Horizontal control-depth view: each decision layer folds routing uncertainty toward zero.
      </Translate>
    ),
  },
]

const FIGURE1_MAPPING = [
  {
    shannon: translate({ id: 'homepage.paperFigures.figure1.mapping.source.shannon', message: 'Source' }),
    vsr: translate({ id: 'homepage.paperFigures.figure1.mapping.source.vsr', message: 'Query r' }),
    note: translate({
      id: 'homepage.paperFigures.figure1.mapping.source.note',
      message: 'The user request is the raw source message before encoding.',
    }),
  },
  {
    shannon: translate({ id: 'homepage.paperFigures.figure1.mapping.encoder.shannon', message: 'Encoder' }),
    vsr: translate({ id: 'homepage.paperFigures.figure1.mapping.encoder.vsr', message: 'Signal Extraction' }),
    note: translate({
      id: 'homepage.paperFigures.figure1.mapping.encoder.note',
      message: 'Heuristic and encoder models transform text into structured signals.',
    }),
  },
  {
    shannon: translate({ id: 'homepage.paperFigures.figure1.mapping.channel.shannon', message: 'Channel' }),
    vsr: translate({ id: 'homepage.paperFigures.figure1.mapping.channel.vsr', message: 'Signal Vector s' }),
    note: translate({
      id: 'homepage.paperFigures.figure1.mapping.channel.note',
      message: 'The signal vector is the transmission layer between extraction and policy logic.',
    }),
  },
  {
    shannon: translate({ id: 'homepage.paperFigures.figure1.mapping.decoder.shannon', message: 'Decoder' }),
    vsr: translate({ id: 'homepage.paperFigures.figure1.mapping.decoder.vsr', message: 'Decision Engine' }),
    note: translate({
      id: 'homepage.paperFigures.figure1.mapping.decoder.note',
      message: 'Boolean rules decode signal patterns into a deterministic routing decision.',
    }),
  },
  {
    shannon: translate({ id: 'homepage.paperFigures.figure1.mapping.destination.shannon', message: 'Destination' }),
    vsr: translate({ id: 'homepage.paperFigures.figure1.mapping.destination.vsr', message: 'Selected Model' }),
    note: translate({
      id: 'homepage.paperFigures.figure1.mapping.destination.note',
      message: 'A concrete model endpoint is chosen and receives the request.',
    }),
  },
] as const

const FIGURE2_STAGES = [
  {
    short: translate({ id: 'homepage.paperFigures.figure2.stage.raw.short', message: 'Raw' }),
    label: translate({ id: 'homepage.paperFigures.figure2.stage.raw.label', message: 'Raw Query' }),
    value: 96,
    note: translate({
      id: 'homepage.paperFigures.figure2.stage.raw.note',
      message: 'No signals yet: routing entropy is near maximum.',
    }),
  },
  {
    short: translate({ id: 'homepage.paperFigures.figure2.stage.keyword.short', message: '+KW' }),
    label: translate({ id: 'homepage.paperFigures.figure2.stage.keyword.label', message: '+Keyword' }),
    value: 72,
    note: translate({
      id: 'homepage.paperFigures.figure2.stage.keyword.note',
      message: 'Keywords quickly eliminate incompatible model families.',
    }),
  },
  {
    short: translate({ id: 'homepage.paperFigures.figure2.stage.domain.short', message: '+Dom' }),
    label: translate({ id: 'homepage.paperFigures.figure2.stage.domain.label', message: '+Domain' }),
    value: 50,
    note: translate({
      id: 'homepage.paperFigures.figure2.stage.domain.note',
      message: 'Domain classifiers narrow candidate pools with higher precision.',
    }),
  },
  {
    short: translate({ id: 'homepage.paperFigures.figure2.stage.signals.short', message: '+Cpx' }),
    label: translate({ id: 'homepage.paperFigures.figure2.stage.signals.label', message: '+ N x Signals...' }),
    value: 30,
    note: translate({
      id: 'homepage.paperFigures.figure2.stage.signals.note',
      message: 'Complexity, embedding and other signals shrink uncertainty further.',
    }),
  },
  {
    short: translate({ id: 'homepage.paperFigures.figure2.stage.decision.short', message: 'Decision' }),
    label: translate({ id: 'homepage.paperFigures.figure2.stage.decision.label', message: 'Decision' }),
    value: 8,
    note: translate({
      id: 'homepage.paperFigures.figure2.stage.decision.note',
      message: 'Decision rules collapse entropy to a near-deterministic model choice.',
    }),
  },
] as const

type LayerKey = 'input' | 'hidden' | 'projection'

const FIGURE3_LAYERS: Array<{
  key: LayerKey
  tag: string
  title: string
  description: string
  focus: string
}> = [
  {
    key: 'input',
    tag: translate({ id: 'homepage.paperFigures.figure3.layer.input.tag', message: 'Input' }),
    title: translate({ id: 'homepage.paperFigures.figure3.layer.input.title', message: 'Signal Extraction' }),
    description: translate({
      id: 'homepage.paperFigures.figure3.layer.input.description',
      message: 'Heuristic + encoder signals produce structured features.',
    }),
    focus: translate({
      id: 'homepage.paperFigures.figure3.layer.input.focus',
      message: 'Collect request features in parallel and emit typed signal values.',
    }),
  },
  {
    key: 'hidden',
    tag: translate({ id: 'homepage.paperFigures.figure3.layer.hidden.tag', message: 'Hidden' }),
    title: translate({ id: 'homepage.paperFigures.figure3.layer.hidden.title', message: 'Decision Blocks' }),
    description: translate({
      id: 'homepage.paperFigures.figure3.layer.hidden.description',
      message: 'Boolean policies choose one scoped routing decision.',
    }),
    focus: translate({
      id: 'homepage.paperFigures.figure3.layer.hidden.focus',
      message: 'Evaluate rule trees and collapse candidates into one active route.',
    }),
  },
  {
    key: 'projection',
    tag: translate({ id: 'homepage.paperFigures.figure3.layer.projection.tag', message: 'Projection' }),
    title: translate({ id: 'homepage.paperFigures.figure3.layer.projection.title', message: 'Plugin Chain' }),
    description: translate({
      id: 'homepage.paperFigures.figure3.layer.projection.description',
      message: 'Pre/post plugins and model selection execute per decision.',
    }),
    focus: translate({
      id: 'homepage.paperFigures.figure3.layer.projection.focus',
      message: 'Apply policy-bound plugins, then dispatch to the selected endpoint.',
    }),
  },
]

type SignalGroupKey = 'heuristic' | 'learned'

const FIGURE4_GROUPS: Record<SignalGroupKey, { title: string, note: string, signals: string[] }> = {
  heuristic: {
    title: translate({ id: 'homepage.paperFigures.figure4.group.heuristic.title', message: 'Heuristic (<1ms)' }),
    note: translate({
      id: 'homepage.paperFigures.figure4.group.heuristic.note',
      message: 'Cheap deterministic filters run first for early pruning.',
    }),
    signals: [
      translate({ id: 'homepage.paperFigures.figure4.signal.keyword', message: 'Keyword' }),
      translate({ id: 'homepage.paperFigures.figure4.signal.language', message: 'Language' }),
      translate({ id: 'homepage.paperFigures.figure4.signal.context', message: 'Context' }),
      translate({ id: 'homepage.paperFigures.figure4.signal.authz', message: 'Authz' }),
    ],
  },
  learned: {
    title: translate({ id: 'homepage.paperFigures.figure4.group.learned.title', message: 'Learned (10-100ms)' }),
    note: translate({
      id: 'homepage.paperFigures.figure4.group.learned.note',
      message: 'Model-based signals add semantic precision for final routing confidence.',
    }),
    signals: [
      translate({ id: 'homepage.paperFigures.figure4.signal.embedding', message: 'Embedding' }),
      translate({ id: 'homepage.paperFigures.figure4.signal.domain', message: 'Domain' }),
      translate({ id: 'homepage.paperFigures.figure4.signal.factual', message: 'Factual' }),
      translate({ id: 'homepage.paperFigures.figure4.signal.feedback', message: 'Feedback' }),
      translate({ id: 'homepage.paperFigures.figure4.signal.modality', message: 'Modality' }),
      translate({ id: 'homepage.paperFigures.figure4.signal.complexity', message: 'Complexity' }),
      translate({ id: 'homepage.paperFigures.figure4.signal.jailbreak', message: 'Jailbreak' }),
      translate({ id: 'homepage.paperFigures.figure4.signal.pii', message: 'PII' }),
      translate({ id: 'homepage.paperFigures.figure4.signal.preference', message: 'Preference' }),
    ],
  },
}

type Figure14NodeKey = 'spec' | 'agent' | 'dsl' | 'engine' | 'feedback'

const FIGURE14_MAIN_FLOW = [
  {
    key: 'spec',
    label: translate({ id: 'homepage.paperFigures.figure14.flow.spec.label', message: 'Natural-Language Spec' }),
    detail: translate({
      id: 'homepage.paperFigures.figure14.flow.spec.detail',
      message: '"Route math and enforce PII policy"',
    }),
    edge: translate({ id: 'homepage.paperFigures.figure14.flow.spec.edge', message: 'synthesis' }),
  },
  {
    key: 'agent',
    label: translate({ id: 'homepage.paperFigures.figure14.flow.agent.label', message: 'Coding Agent (LLM)' }),
    detail: translate({
      id: 'homepage.paperFigures.figure14.flow.agent.detail',
      message: 'Policy compiler over constrained DSL',
    }),
    edge: translate({ id: 'homepage.paperFigures.figure14.flow.agent.edge', message: 'generate' }),
  },
  {
    key: 'dsl',
    label: translate({ id: 'homepage.paperFigures.figure14.flow.dsl.label', message: 'Neural-symbolic DSL' }),
    detail: translate({ id: 'homepage.paperFigures.figure14.flow.dsl.detail', message: 'Typed route program' }),
    edge: translate({ id: 'homepage.paperFigures.figure14.flow.dsl.edge', message: 'instantiate' }),
  },
  {
    key: 'engine',
    label: translate({ id: 'homepage.paperFigures.figure14.flow.engine.label', message: 'Semantics Engine' }),
    detail: translate({
      id: 'homepage.paperFigures.figure14.flow.engine.detail',
      message: 'Signal-decision-plugin execution',
    }),
    edge: '',
  },
] as const

const FIGURE14_FEEDBACK = {
  key: 'feedback' as const,
  label: translate({
    id: 'homepage.paperFigures.figure14.feedback.label',
    message: 'Routing Quality Feedback Q(r, m*)',
  }),
  detail: translate({
    id: 'homepage.paperFigures.figure14.feedback.detail',
    message: 'Optimize synthesis policy from runtime outcomes',
  }),
}

const FIGURE14_NOTES: Record<Figure14NodeKey, string> = {
  spec: translate({
    id: 'homepage.paperFigures.figure14.note.spec',
    message: 'Natural-language requirements define routing intent before policy synthesis starts.',
  }),
  agent: translate({
    id: 'homepage.paperFigures.figure14.note.agent',
    message: 'The coding agent translates intent into executable DSL policy with constrained syntax.',
  }),
  dsl: translate({
    id: 'homepage.paperFigures.figure14.note.dsl',
    message: 'The generated DSL config is auditable and compiles to deterministic routing behavior.',
  }),
  engine: translate({
    id: 'homepage.paperFigures.figure14.note.engine',
    message: 'Inference executes policy over signal extraction, decision evaluation, and plugin projection.',
  }),
  feedback: translate({
    id: 'homepage.paperFigures.figure14.note.feedback',
    message: 'Runtime quality closes the loop and tunes the next synthesis iteration.',
  }),
}

type Figure9LayerKey = 'input' | 'embed' | 'hidden' | 'select' | 'plugins'
type Figure9LayerKind = 'io' | 'decision' | 'head' | 'projection'

const FIGURE9_TRANSFORMER_LAYERS: Array<{
  key: Figure9LayerKey
  tag: string
  title: string
  formula: string
  uncertainty: number
  uLabel: string
  kind: Figure9LayerKind
  note: string
}> = [
  {
    key: 'input',
    tag: translate({ id: 'homepage.paperFigures.figure9.layer.input.tag', message: 'Input' }),
    title: translate({ id: 'homepage.paperFigures.figure9.layer.input.title', message: 'Input Query r' }),
    formula: translate({ id: 'homepage.paperFigures.figure9.layer.input.formula', message: 'raw request' }),
    uncertainty: 100,
    uLabel: 'U0',
    kind: 'io',
    note: translate({
      id: 'homepage.paperFigures.figure9.layer.input.note',
      message: 'Routing starts at maximal uncertainty before any control signals are extracted.',
    }),
  },
  {
    key: 'embed',
    tag: translate({ id: 'homepage.paperFigures.figure9.layer.embed.tag', message: 'Embed' }),
    title: translate({ id: 'homepage.paperFigures.figure9.layer.embed.title', message: 'Hybrid Signal Embedding' }),
    formula: 'S(r) = s',
    uncertainty: 84,
    uLabel: 'U1',
    kind: 'io',
    note: translate({
      id: 'homepage.paperFigures.figure9.layer.embed.note',
      message: 'Signal extraction creates a shared control state that all downstream layers read.',
    }),
  },
  {
    key: 'hidden',
    tag: translate({ id: 'homepage.paperFigures.figure9.layer.hidden.tag', message: 'Hidden' }),
    title: translate({
      id: 'homepage.paperFigures.figure9.layer.hidden.title',
      message: 'Decision Hidden Layers (x N)',
    }),
    formula: 'φ1(s) ... φN(s) → z*',
    uncertainty: 34,
    uLabel: 'UL',
    kind: 'decision',
    note: translate({
      id: 'homepage.paperFigures.figure9.layer.hidden.note',
      message: 'A stacked hidden control depth is represented as one block that folds entropy across N decision layers.',
    }),
  },
  {
    key: 'select',
    tag: translate({ id: 'homepage.paperFigures.figure9.layer.select.tag', message: 'Head' }),
    title: translate({ id: 'homepage.paperFigures.figure9.layer.select.title', message: 'Selection Head' }),
    formula: 'priority early-exit ⇒ (d*, m*)',
    uncertainty: 10,
    uLabel: '~0',
    kind: 'head',
    note: translate({
      id: 'homepage.paperFigures.figure9.layer.select.note',
      message: 'Selection head resolves the final decision with deterministic early-exit behavior.',
    }),
  },
  {
    key: 'plugins',
    tag: translate({ id: 'homepage.paperFigures.figure9.layer.plugins.tag', message: 'Projection' }),
    title: translate({ id: 'homepage.paperFigures.figure9.layer.plugins.title', message: 'Plugin Projection' }),
    formula: 'Ψd* = πn ∘ ... ∘ π1',
    uncertainty: 6,
    uLabel: 'resolved',
    kind: 'projection',
    note: translate({
      id: 'homepage.paperFigures.figure9.layer.plugins.note',
      message: 'Projection head applies policy-bound plugins after route selection is finalized.',
    }),
  },
]

const Figure1Panel: React.FC = () => {
  const [activeColumn, setActiveColumn] = useState(0)
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { margin: '-60px' })

  useEffect(() => {
    if (!inView) return
    const timer = setInterval(() => {
      setActiveColumn(prev => (prev + 1) % FIGURE1_MAPPING.length)
    }, 2200)
    return () => clearInterval(timer)
  }, [inView])

  return (
    <div ref={ref} className={`${styles.figureCanvas} ${styles.figureCanvasFill}`}>
      <div className={styles.mappingGrid}>
        {FIGURE1_MAPPING.map((item, idx) => {
          const isActive = idx === activeColumn
          return (
            <motion.button
              key={item.shannon}
              type="button"
              className={`${styles.mappingColumn} ${isActive ? styles.mappingColumnActive : ''}`}
              onMouseEnter={() => setActiveColumn(idx)}
              onFocus={() => setActiveColumn(idx)}
              onClick={() => setActiveColumn(idx)}
              aria-pressed={isActive}
              animate={inView
                ? { opacity: isActive ? 1 : 0.55, y: 0, scale: isActive ? 1 : 0.98 }
                : { opacity: 0.45, y: 8, scale: 0.98 }}
              transition={{ duration: 0.25 }}
              whileHover={{ scale: 1.01 }}
            >
              <div className={styles.mappingSubBlock}>
                <span className={styles.mappingSideLabel}>
                  <Translate id="homepage.paperFigures.figure1.mappingSide.shannon">Shannon</Translate>
                </span>
                <span className={styles.mappingNode}>{item.shannon}</span>
              </div>
              <span className={styles.mappingBridge}>↕</span>
              <div className={`${styles.mappingSubBlock} ${styles.mappingSubBlockSecondary}`}>
                <span className={styles.mappingSideLabel}>
                  <Translate id="homepage.paperFigures.figure1.mappingSide.vsr">VSR</Translate>
                </span>
                <span className={styles.mappingNode}>{item.vsr}</span>
              </div>
            </motion.button>
          )
        })}
      </div>
      <p className={styles.mappingNote}>{FIGURE1_MAPPING[activeColumn].note}</p>
    </div>
  )
}

const Figure2Panel: React.FC = () => {
  const [activeStage, setActiveStage] = useState(0)
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { margin: '-60px' })

  useEffect(() => {
    if (!inView) return
    setActiveStage(0)
    const timer = setInterval(() => {
      setActiveStage(prev => (prev + 1) % FIGURE2_STAGES.length)
    }, 1800)
    return () => clearInterval(timer)
  }, [inView])

  return (
    <div ref={ref} className={styles.figureCanvas}>
      <div className={styles.entropyStageRow}>
        {FIGURE2_STAGES.map((stage, idx) => {
          const isActive = idx === activeStage
          return (
            <button
              key={stage.label}
              type="button"
              className={`${styles.entropyStageChip} ${isActive ? styles.entropyStageChipActive : ''}`}
              onClick={() => setActiveStage(idx)}
            >
              {stage.short}
            </button>
          )
        })}
      </div>
      <div className={styles.entropyGrid}>
        {FIGURE2_STAGES.map((bar, i) => {
          const isActive = i === activeStage
          return (
            <div key={bar.label} className={styles.entropyBarWrap}>
              <motion.div
                className={styles.entropyBar}
                animate={inView
                  ? { height: `${bar.value}%`, opacity: isActive ? 1 : 0.55, scaleX: isActive ? 1 : 0.93 }
                  : { height: '0%', opacity: 0.4, scaleX: 0.9 }}
                transition={{ delay: i * 0.08, duration: 0.35 }}
              />
              <span className={`${styles.entropyLabel} ${isActive ? styles.entropyLabelActive : ''}`}>
                {bar.label}
              </span>
            </div>
          )
        })}
      </div>
      <p className={styles.entropyNote}>{FIGURE2_STAGES[activeStage].note}</p>
    </div>
  )
}

const Figure3Panel: React.FC = () => {
  const [activeLayer, setActiveLayer] = useState<LayerKey>('input')
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { margin: '-60px' })

  useEffect(() => {
    if (!inView) return
    const timer = setInterval(() => {
      setActiveLayer((prev) => {
        const index = FIGURE3_LAYERS.findIndex(layer => layer.key === prev)
        const nextIndex = (index + 1) % FIGURE3_LAYERS.length
        return FIGURE3_LAYERS[nextIndex].key
      })
    }, 2400)
    return () => clearInterval(timer)
  }, [inView])

  const activeLayerMeta = FIGURE3_LAYERS.find(layer => layer.key === activeLayer) ?? FIGURE3_LAYERS[0]

  return (
    <div ref={ref} className={`${styles.figureCanvas} ${styles.figureCanvasFill}`}>
      <div className={styles.layerGrid}>
        {FIGURE3_LAYERS.map((layer, index) => {
          const isActive = layer.key === activeLayer
          return (
            <React.Fragment key={layer.key}>
              <motion.button
                type="button"
                className={`${styles.layerCard} ${isActive ? styles.layerCardActive : ''}`}
                onClick={() => setActiveLayer(layer.key)}
                onFocus={() => setActiveLayer(layer.key)}
                animate={inView
                  ? { opacity: isActive ? 1 : 0.6, y: 0, scale: isActive ? 1 : 0.98 }
                  : { opacity: 0.45, y: 8, scale: 0.98 }}
                transition={{ duration: 0.25 }}
                whileHover={{ scale: 1.01 }}
              >
                <span className={styles.layerTag}>{layer.tag}</span>
                <h4 className={styles.layerTitle}>{layer.title}</h4>
                <p className={styles.layerText}>{layer.description}</p>
              </motion.button>
              {index < FIGURE3_LAYERS.length - 1 && <div className={styles.layerConnector}>→</div>}
            </React.Fragment>
          )
        })}
      </div>

      <div className={styles.layerDetailCard}>
        <span className={styles.layerDetailKicker}>
          {activeLayerMeta.tag}
          {' '}
          <Translate id="homepage.paperFigures.figure3.layerFocus">Layer Focus</Translate>
        </span>
        <p className={styles.layerDetailText}>{activeLayerMeta.focus}</p>
      </div>
    </div>
  )
}

const Figure4Panel: React.FC = () => {
  const [activeGroup, setActiveGroup] = useState<SignalGroupKey>('heuristic')
  const [activeSignal, setActiveSignal] = useState(FIGURE4_GROUPS.heuristic.signals[0])
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { margin: '-60px' })

  useEffect(() => {
    if (!inView) return
    const timer = setInterval(() => {
      setActiveGroup(prev => (prev === 'heuristic' ? 'learned' : 'heuristic'))
    }, 2600)
    return () => clearInterval(timer)
  }, [inView])

  useEffect(() => {
    setActiveSignal(FIGURE4_GROUPS[activeGroup].signals[0])
  }, [activeGroup])

  return (
    <div ref={ref} className={`${styles.figureCanvas} ${styles.figureCanvasFill}`}>
      <div className={styles.taxonomyGrid}>
        {(['heuristic', 'learned'] as SignalGroupKey[]).map((group, index) => {
          const groupMeta = FIGURE4_GROUPS[group]
          const isActiveGroup = group === activeGroup
          return (
            <React.Fragment key={group}>
              <motion.div
                className={`${styles.taxonomyCard} ${isActiveGroup ? styles.taxonomyCardActive : styles.taxonomyCardMuted}`}
                onClick={() => setActiveGroup(group)}
                onMouseEnter={() => setActiveGroup(group)}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    setActiveGroup(group)
                  }
                }}
                animate={inView
                  ? { opacity: isActiveGroup ? 1 : 0.62, y: 0, scale: isActiveGroup ? 1 : 0.98 }
                  : { opacity: 0.45, y: 8, scale: 0.98 }}
                transition={{ duration: 0.25 }}
              >
                <span className={styles.taxonomyTitle}>{groupMeta.title}</span>
                <div className={styles.tokenRow}>
                  {groupMeta.signals.map((signal) => {
                    const isActiveSignal = isActiveGroup && signal === activeSignal
                    return (
                      <button
                        key={signal}
                        type="button"
                        className={`${styles.tokenTag} ${isActiveSignal ? styles.tokenTagActive : styles.tokenTagMuted}`}
                        onMouseEnter={() => setActiveSignal(signal)}
                        onFocus={() => setActiveSignal(signal)}
                        onClick={() => setActiveSignal(signal)}
                      >
                        {signal}
                      </button>
                    )
                  })}
                </div>
              </motion.div>
              {index === 0 && <div className={styles.taxonomyArrow}>+</div>}
            </React.Fragment>
          )
        })}
        <div className={styles.taxonomyArrow}>→</div>
        <div className={styles.taxonomyResult}>S(r)</div>
      </div>
      <p className={styles.taxonomyNote}>
        {translate({
          id: 'homepage.paperFigures.figure4.summaryText',
          message: '13 total signals. Active signal: {signal}. {groupNote}',
          values: {
            signal: activeSignal,
            groupNote: FIGURE4_GROUPS[activeGroup].note,
          },
        })}
      </p>
    </div>
  )
}

const Figure14Panel: React.FC = () => {
  const [activeNode, setActiveNode] = useState<Figure14NodeKey>('spec')
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { margin: '-60px' })

  const rotationOrder: Figure14NodeKey[] = useMemo(
    () => [...FIGURE14_MAIN_FLOW.map(step => step.key), FIGURE14_FEEDBACK.key],
    [],
  )

  useEffect(() => {
    if (!inView) return
    const timer = setInterval(() => {
      setActiveNode((prev) => {
        const index = rotationOrder.findIndex(key => key === prev)
        const nextIndex = (index + 1) % rotationOrder.length
        return rotationOrder[nextIndex]
      })
    }, 2400)
    return () => clearInterval(timer)
  }, [inView, rotationOrder])

  return (
    <div ref={ref} className={styles.figureCanvas}>
      <div className={styles.agentMainFlow}>
        {FIGURE14_MAIN_FLOW.map((step, index) => {
          const isActive = step.key === activeNode
          const isSpecNode = step.key === 'spec'
          const isAgentNode = step.key === 'agent'
          const isDslNode = step.key === 'dsl'
          const isEngineNode = step.key === 'engine'
          return (
            <React.Fragment key={step.key}>
              <motion.button
                type="button"
                className={`${styles.agentNode} ${isActive ? styles.agentNodeActive : styles.agentNodeMuted} ${isAgentNode ? styles.agentNodeAgent : ''} ${isEngineNode ? styles.agentNodeEngine : ''}`}
                onClick={() => setActiveNode(step.key)}
                onFocus={() => setActiveNode(step.key)}
                animate={inView
                  ? { opacity: isActive ? 1 : 0.62, y: 0, scale: isActive ? 1 : 0.98 }
                  : { opacity: 0.45, y: 8, scale: 0.98 }}
                transition={{ duration: 0.22 }}
              >
                {isSpecNode && (
                  <span className={styles.agentSpeaker} aria-hidden="true">
                    <svg viewBox="0 0 64 64" className={styles.agentRobotSvg}>
                      <circle cx="22" cy="23" r="8" fill="none" stroke="currentColor" strokeWidth="4" />
                      <path d="M9 45c1-7 6-12 13-12s12 5 13 12" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
                      <path d="M39 25c5 0 9 4 9 9" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
                      <path d="M43 18c8 0 14 6 14 14" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
                    </svg>
                  </span>
                )}
                {isAgentNode && (
                  <span className={styles.agentRobot} aria-hidden="true">
                    <svg viewBox="0 0 64 64" className={styles.agentRobotSvg}>
                      <rect x="16" y="18" width="32" height="30" rx="9" fill="none" stroke="currentColor" strokeWidth="4" />
                      <circle cx="27" cy="32" r="3" fill="currentColor" />
                      <circle cx="37" cy="32" r="3" fill="currentColor" />
                      <path d="M24 40h16" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
                      <path d="M32 10v6" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
                      <circle cx="32" cy="8" r="3" fill="currentColor" />
                      <path d="M16 30h-5M53 30h-5" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
                    </svg>
                  </span>
                )}
                {isDslNode && (
                  <span className={styles.agentCode} aria-hidden="true">
                    <svg viewBox="0 0 64 64" className={styles.agentRobotSvg}>
                      <rect x="10" y="14" width="44" height="36" rx="8" fill="none" stroke="currentColor" strokeWidth="4" />
                      <path d="M25 27l-8 7 8 7" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
                      <path d="M39 27l8 7-8 7" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
                      <path d="M33 24l-4 20" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
                    </svg>
                  </span>
                )}
                {isEngineNode && (
                  <span className={styles.agentEngine} aria-hidden="true">
                    <svg viewBox="0 0 64 64" className={styles.agentRobotSvg}>
                      <circle cx="32" cy="32" r="8" fill="none" stroke="currentColor" strokeWidth="4" />
                      <circle cx="32" cy="32" r="2.5" fill="currentColor" />
                      <path d="M32 12v7M32 45v7M12 32h7M45 32h7" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
                      <path d="M18 18l5 5M41 41l5 5M46 18l-5 5M23 41l-5 5" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
                    </svg>
                  </span>
                )}
                <span className={styles.agentNodeLabel}>{step.label}</span>
                <span className={styles.agentNodeDetail}>{step.detail}</span>
                {isEngineNode && (
                  <span className={styles.agentNodeOutput}>
                    <Translate id="homepage.paperFigures.figure14.engineOutput">Engine Output m*</Translate>
                  </span>
                )}
              </motion.button>
              {index < FIGURE14_MAIN_FLOW.length - 1 && (
                <div className={styles.agentFlowEdge}>
                  <span className={styles.agentFlowArrow}>→</span>
                  <span className={styles.agentFlowVerb}>{step.edge}</span>
                </div>
              )}
            </React.Fragment>
          )
        })}
      </div>

      <div className={styles.agentLoopTrack}>
        <div className={styles.agentLoopDrop}>
          <span className={styles.agentLoopDropArrow}>↓</span>
          <span className={styles.agentLoopDropText}>
            <Translate id="homepage.paperFigures.figure14.loop.engineOutput">Engine Output</Translate>
          </span>
        </div>
        <button
          type="button"
          className={`${styles.agentFeedbackCard} ${activeNode === 'feedback' ? styles.agentFeedbackCardActive : ''}`}
          onClick={() => setActiveNode('feedback')}
        >
          <span className={styles.agentFeedbackTitle}>{FIGURE14_FEEDBACK.label}</span>
          <span className={styles.agentFeedbackDetail}>{FIGURE14_FEEDBACK.detail}</span>
        </button>
        <div className={styles.agentLoopReturn}>
          <span className={styles.agentLoopReturnArrow}>↺</span>
          <span className={styles.agentLoopReturnText}>
            <Translate id="homepage.paperFigures.figure14.loop.optimize">optimize → Coding Agent</Translate>
          </span>
        </div>
      </div>

      <p className={styles.agentFlowNote}>{FIGURE14_NOTES[activeNode]}</p>
    </div>
  )
}

const Figure9Panel: React.FC = () => {
  const [activeLayer, setActiveLayer] = useState(0)
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { margin: '-60px' })
  const decisionLayerIndices = FIGURE9_TRANSFORMER_LAYERS.reduce<number[]>((acc, layer, index) => {
    if (layer.kind === 'decision') acc.push(index)
    return acc
  }, [])

  useEffect(() => {
    if (!inView) return
    setActiveLayer(0)
    const timer = setInterval(() => {
      setActiveLayer(prev => (prev + 1) % FIGURE9_TRANSFORMER_LAYERS.length)
    }, 1900)
    return () => clearInterval(timer)
  }, [inView])

  return (
    <div ref={ref} className={styles.figureCanvas}>
      <div className={styles.transformerBus}>
        <span className={styles.transformerBusLabel}>
          <Translate id="homepage.paperFigures.figure9.busLabel">
            Shared control state s feeds each decision layer
          </Translate>
        </span>
      </div>

      <div className={styles.transformerStackRow}>
        {FIGURE9_TRANSFORMER_LAYERS.map((layer, index) => {
          const isActive = index === activeLayer
          const isDecisionLayer = layer.kind === 'decision'
          return (
            <React.Fragment key={layer.key}>
              <motion.button
                type="button"
                className={`${styles.transformerBlock} ${isActive ? styles.transformerBlockActive : styles.transformerBlockMuted} ${isDecisionLayer ? styles.transformerBlockDecision : ''}`}
                onClick={() => setActiveLayer(index)}
                animate={inView
                  ? { opacity: isActive ? 1 : 0.64, y: 0, scale: isActive ? 1 : 0.98 }
                  : { opacity: 0.45, y: 8, scale: 0.98 }}
                transition={{ duration: 0.24 }}
              >
                <span className={styles.transformerBlockTag}>{layer.tag}</span>
                <span className={styles.transformerBlockTitle}>{layer.title}</span>
                <span className={styles.transformerBlockFormula}>{layer.formula}</span>
                {isDecisionLayer && (
                  <div className={styles.transformerDepthBars}>
                    <span />
                    <span />
                    <span />
                  </div>
                )}
                <div className={styles.transformerEntropyRow}>
                  <span className={styles.transformerEntropyLabel}>{layer.uLabel}</span>
                  <div className={styles.transformerEntropyTrack}>
                    <motion.div
                      className={styles.transformerEntropyBar}
                      animate={inView ? { width: `${layer.uncertainty}%` } : { width: '0%' }}
                      transition={{ duration: 0.3, delay: index * 0.05 }}
                    />
                  </div>
                </div>
              </motion.button>
              {index < FIGURE9_TRANSFORMER_LAYERS.length - 1 && <span className={styles.transformerArrow}>→</span>}
            </React.Fragment>
          )
        })}
      </div>

      <div className={styles.transformerExitLane}>
        <span className={styles.transformerExitLabel}>
          <Translate id="homepage.paperFigures.figure9.exitLabel">
            Priority early-exit taps from control layers
          </Translate>
        </span>
        <div className={styles.transformerExitDots}>
          {decisionLayerIndices.map(index => (
            <span
              key={index}
              className={`${styles.transformerExitDot} ${activeLayer === index ? styles.transformerExitDotActive : ''}`}
            >
              {FIGURE9_TRANSFORMER_LAYERS[index].tag}
            </span>
          ))}
        </div>
      </div>

      <p className={styles.foldNote}>{FIGURE9_TRANSFORMER_LAYERS[activeLayer].note}</p>
    </div>
  )
}

const panelMap: Record<FigureKey, React.FC> = {
  figure1: Figure1Panel,
  figure2: Figure2Panel,
  figure3: Figure3Panel,
  figure4: Figure4Panel,
  figure14: Figure14Panel,
  figure9: Figure9Panel,
}

const PaperFigureShowcase: React.FC = () => {
  const [activeFigure, setActiveFigure] = useState<FigureKey>('figure1')

  const activeMeta = useMemo(
    () => FIGURES.find(figure => figure.key === activeFigure) ?? FIGURES[0],
    [activeFigure],
  )

  const ActivePanel = panelMap[activeFigure]

  return (
    <section className={styles.paperFigureSection}>
      <div className="container">
        <div className={styles.paperHeader}>
          <p className={styles.paperLabel}>
            <Translate id="homepage.paperFigures.label">Routing Blueprint</Translate>
          </p>
          <h2 className={styles.paperTitle}>
            <Translate id="homepage.paperFigures.title">How System Works</Translate>
          </h2>
          <p className={styles.paperDescription}>
            <Translate id="homepage.paperFigures.description">
              An interactive walkthrough of signal extraction, decision logic, and model routing behavior.
            </Translate>
          </p>
        </div>

        <div className={styles.paperWorkspace}>
          <div className={styles.figureNav}>
            {FIGURES.map(figure => (
              <button
                key={figure.key}
                className={`${styles.figureNavItem} ${figure.key === activeFigure ? styles.figureNavItemActive : ''}`}
                onClick={() => setActiveFigure(figure.key)}
                type="button"
              >
                <span className={styles.figureIndex}>{figure.index}</span>
                <span className={styles.figureNavText}>{figure.title}</span>
              </button>
            ))}
          </div>

          <div className={styles.figureStage}>
            <div className={styles.figureStageHeader}>
              <h3 className={styles.figureStageTitle}>{activeMeta.title}</h3>
              <p className={styles.figureStageSummary}>{activeMeta.summary}</p>
            </div>
            <ActivePanel />
          </div>
        </div>
      </div>
    </section>
  )
}

export default PaperFigureShowcase
