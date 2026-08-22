import React, { useState } from 'react'
import Link from '@docusaurus/Link'
import Translate, { translate } from '@docusaurus/Translate'
import { PillLink } from '@site/src/components/site/Chrome'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import shared from './homepageShared.module.css'
import styles from './UseCaseExplorer.module.css'

type UseCaseFeature = {
  title: string
  description: string
  docTo: string
}

type FlowStep = {
  label: string
  detail: string
}

type UseCaseTab = {
  id: string
  label: string
  subtitle: string
  heading: string
  summary: string
  flow: FlowStep[]
  features: UseCaseFeature[]
}

const useCaseTabs: UseCaseTab[] = [
  {
    id: 'semantic-routing',
    label: translate({
      id: 'homepage.useCases.tab.routing.label',
      message: 'Semantic Routing',
    }),
    subtitle: translate({
      id: 'homepage.useCases.tab.routing.subtitle',
      message: 'Prompt-aware model selection',
    }),
    heading: translate({
      id: 'homepage.useCases.tab.routing.heading',
      message: 'Semantic Routing',
    }),
    summary: translate({
      id: 'homepage.useCases.tab.routing.summary',
      message:
        'Classify request signals, apply recipe policy, and select an eligible model from the configured pool.',
    }),
    flow: [
      {
        label: translate({ id: 'homepage.useCases.tab.routing.flow1', message: 'Client request' }),
        detail: translate({ id: 'homepage.useCases.tab.routing.flow1.detail', message: 'OpenAI-compatible API' }),
      },
      {
        label: translate({ id: 'homepage.useCases.tab.routing.flow2', message: 'Signal layer' }),
        detail: translate({ id: 'homepage.useCases.tab.routing.flow2.detail', message: '20 signal families' }),
      },
      {
        label: translate({ id: 'homepage.useCases.tab.routing.flow3', message: 'Decision engine' }),
        detail: translate({ id: 'homepage.useCases.tab.routing.flow3.detail', message: 'router_dc · multi_factor' }),
      },
      {
        label: translate({ id: 'homepage.useCases.tab.routing.flow4', message: 'Model pool' }),
        detail: translate({ id: 'homepage.useCases.tab.routing.flow4.detail', message: 'Target model' }),
      },
    ],
    features: [
      {
        title: translate({
          id: 'homepage.useCases.tab.routing.feature1.title',
          message: 'Route by signal, not just load',
        }),
        description: translate({
          id: 'homepage.useCases.tab.routing.feature1.desc',
          message:
            'Combine embeddings, domain, PII, jailbreak, preference, and more into executable routing decisions.',
        }),
        docTo: '/docs/tutorials/signal/overview',
      },
      {
        title: translate({
          id: 'homepage.useCases.tab.routing.feature2.title',
          message: 'Specialist pools with router_dc',
        }),
        description: translate({
          id: 'homepage.useCases.tab.routing.feature2.desc',
          message:
            'Match prompts to model descriptions with embedding similarity for Cursor-style Auto routing.',
        }),
        docTo: '/docs/tutorials/algorithm/selection/router-dc',
      },
      {
        title: translate({
          id: 'homepage.useCases.tab.routing.feature3.title',
          message: 'Operational tradeoffs with multi_factor',
        }),
        description: translate({
          id: 'homepage.useCases.tab.routing.feature3.desc',
          message:
            'Balance quality, latency, cost, and load using model metadata and live local metrics.',
        }),
        docTo: '/docs/tutorials/algorithm/selection/multi-factor',
      },
      {
        title: translate({
          id: 'homepage.useCases.tab.routing.feature4.title',
          message: 'Cascade and fuse across models',
        }),
        description: translate({
          id: 'homepage.useCases.tab.routing.feature4.desc',
          message:
            'Use loopers to triage with lightweight models, escalate difficult prompts, or combine answers.',
        }),
        docTo: '/docs/intro',
      },
    ],
  },
  {
    id: 'policy-guardrails',
    label: translate({
      id: 'homepage.useCases.tab.policy.label',
      message: 'Policy & Guardrails',
    }),
    subtitle: translate({
      id: 'homepage.useCases.tab.policy.subtitle',
      message: 'Inline safety and access control',
    }),
    heading: translate({
      id: 'homepage.useCases.tab.policy.heading',
      message: 'Policy & Guardrails',
    }),
    summary: translate({
      id: 'homepage.useCases.tab.policy.summary',
      message:
        'Evaluate configured PII, jailbreak, authorization, and rate-limit policy before model dispatch.',
    }),
    flow: [
      {
        label: translate({ id: 'homepage.useCases.tab.policy.flow1', message: 'Incoming request' }),
        detail: translate({ id: 'homepage.useCases.tab.policy.flow1.detail', message: 'Headers + body' }),
      },
      {
        label: translate({ id: 'homepage.useCases.tab.policy.flow2', message: 'Classifiers' }),
        detail: translate({ id: 'homepage.useCases.tab.policy.flow2.detail', message: 'PII · jailbreak' }),
      },
      {
        label: translate({ id: 'homepage.useCases.tab.policy.flow3', message: 'Policy layer' }),
        detail: translate({ id: 'homepage.useCases.tab.policy.flow3.detail', message: 'Authz · rate limits' }),
      },
      {
        label: translate({ id: 'homepage.useCases.tab.policy.flow4', message: 'Action' }),
        detail: translate({ id: 'homepage.useCases.tab.policy.flow4.detail', message: 'Allow · block · route' }),
      },
    ],
    features: [
      {
        title: translate({
          id: 'homepage.useCases.tab.policy.feature1.title',
          message: 'Detect prompt attacks and sensitive data',
        }),
        description: translate({
          id: 'homepage.useCases.tab.policy.feature1.desc',
          message:
            'Run jailbreak and PII classifiers inline, then block or steer requests through configured decisions.',
        }),
        docTo: '/docs/tutorials/signal/learned/pii',
      },
      {
        title: translate({
          id: 'homepage.useCases.tab.policy.feature2.title',
          message: 'Authz-aware routing',
        }),
        description: translate({
          id: 'homepage.useCases.tab.policy.feature2.desc',
          message:
            'Bind routing decisions to identity and tenant policy for the right users and models.',
        }),
        docTo: '/docs/tutorials/signal/heuristic/authz',
      },
      {
        title: translate({
          id: 'homepage.useCases.tab.policy.feature3.title',
          message: 'Per-route plugin policies',
        }),
        description: translate({
          id: 'homepage.useCases.tab.policy.feature3.desc',
          message:
            'Attach cache, memory, RAG, and hallucination checks per decision — not global config.',
        }),
        docTo: '/docs/tutorials/plugin/overview',
      },
      {
        title: translate({
          id: 'homepage.useCases.tab.policy.feature4.title',
          message: 'Set limits and stop budget surprises',
        }),
        description: translate({
          id: 'homepage.useCases.tab.policy.feature4.desc',
          message:
            'Apply role-based request and token limits, then use pricing metadata in cost-aware selectors.',
        }),
        docTo: '/docs/tutorials/global/api-and-observability',
      },
    ],
  },
  {
    id: 'observability',
    label: translate({
      id: 'homepage.useCases.tab.observability.label',
      message: 'Observability & Replay',
    }),
    subtitle: translate({
      id: 'homepage.useCases.tab.observability.subtitle',
      message: 'Inspect configured routing telemetry',
    }),
    heading: translate({
      id: 'homepage.useCases.tab.observability.heading',
      message: 'Observability & Replay',
    }),
    summary: translate({
      id: 'homepage.useCases.tab.observability.summary',
      message:
        'Enable replay to retain routing evidence for debugging and evaluation. Stored detail, usage, and cost depend on configuration and backend data.',
    }),
    flow: [
      {
        label: translate({ id: 'homepage.useCases.tab.observability.flow1', message: 'Routed request' }),
        detail: translate({ id: 'homepage.useCases.tab.observability.flow1.detail', message: 'Replay-enabled route' }),
      },
      {
        label: translate({ id: 'homepage.useCases.tab.observability.flow2', message: 'Replay record' }),
        detail: translate({ id: 'homepage.useCases.tab.observability.flow2.detail', message: 'Signals + decision' }),
      },
      {
        label: translate({ id: 'homepage.useCases.tab.observability.flow3', message: 'x-vsr-replay-id' }),
        detail: translate({ id: 'homepage.useCases.tab.observability.flow3.detail', message: 'Response header' }),
      },
      {
        label: translate({ id: 'homepage.useCases.tab.observability.flow4', message: 'Dashboard' }),
        detail: translate({ id: 'homepage.useCases.tab.observability.flow4.detail', message: 'Analytics + OTel' }),
      },
    ],
    features: [
      {
        title: translate({
          id: 'homepage.useCases.tab.observability.feature1.title',
          message: 'Inspect routing records',
        }),
        description: translate({
          id: 'homepage.useCases.tab.observability.feature1.desc',
          message:
            'Replay can store decision metadata plus usage and cost when providers and recipe settings supply them.',
        }),
        docTo: '/docs/tutorials/global/api-and-observability',
      },
      {
        title: translate({
          id: 'homepage.useCases.tab.observability.feature2.title',
          message: 'Dashboard insights and analytics',
        }),
        description: translate({
          id: 'homepage.useCases.tab.observability.feature2.desc',
          message:
            'Explore decision distribution and signal frequency from available replay and metrics data.',
        }),
        docTo: '/docs/installation',
      },
      {
        title: translate({
          id: 'homepage.useCases.tab.observability.feature3.title',
          message: 'Correlate with x-vsr-replay-id',
        }),
        description: translate({
          id: 'homepage.useCases.tab.observability.feature3.desc',
          message:
            'Replay-enabled responses can carry an ID that links operators to the corresponding routing record.',
        }),
        docTo: '/docs/tutorials/plugin/router-replay',
      },
      {
        title: translate({
          id: 'homepage.useCases.tab.observability.feature4.title',
          message: 'OpenTelemetry-ready operations',
        }),
        description: translate({
          id: 'homepage.useCases.tab.observability.feature4.desc',
          message:
            'Export available routing metrics and health signals for production monitoring.',
        }),
        docTo: '/docs/tutorials/global/api-and-observability',
      },
    ],
  },
]

function UseCaseFlow({ steps }: { steps: FlowStep[] }): JSX.Element {
  return (
    <div className={styles.flowDiagram} aria-hidden="true">
      {steps.map((step, index) => (
        <React.Fragment key={step.label}>
          <div className={styles.flowStep}>
            <span className={styles.flowStepIndex}>{String(index + 1).padStart(2, '0')}</span>
            <div className={styles.flowStepCopy}>
              <strong>{step.label}</strong>
              <span>{step.detail}</span>
            </div>
          </div>
          {index < steps.length - 1 && <span className={styles.flowConnector} />}
        </React.Fragment>
      ))}
    </div>
  )
}

function UseCasePanel({ tab, active }: { tab: UseCaseTab, active: boolean }): JSX.Element {
  return (
    <div
      className={`${styles.panel} ${active ? styles.panelActive : styles.panelInactive}`}
      role="tabpanel"
      id={`use-case-panel-${tab.id}`}
      aria-labelledby={`use-case-tab-${tab.id}`}
      aria-hidden={!active}
    >
      <div className={styles.panelHeader}>
        <div className={styles.panelIntro}>
          <h3>{tab.heading}</h3>
          <p>{tab.summary}</p>
        </div>
      </div>

      <div className={styles.panelBody}>
        <UseCaseFlow steps={tab.flow} />

        <div className={styles.featureList}>
          {tab.features.map((feature, index) => (
            <Link
              key={feature.title}
              className={styles.featureRow}
              to={feature.docTo}
            >
              <span className={styles.featureIndex}>{String(index + 1).padStart(2, '0')}</span>
              <div className={styles.featureCopy}>
                <strong>{feature.title}</strong>
                <p>{feature.description}</p>
              </div>
              <span className={styles.featureArrow} aria-hidden="true">→</span>
            </Link>
          ))}
        </div>
      </div>

      <div className={styles.panelFooter}>
        <div className={styles.integrationRow}>
          <span className={styles.integrationLabel}>
            <Translate id="homepage.useCases.integrations">Integrations</Translate>
          </span>
          <Link className={styles.integrationChip} to="/docs/installation">
            <Translate id="homepage.useCases.integration.extproc">Envoy ExtProc</Translate>
          </Link>
          <Link className={styles.integrationChip} to="/docs/installation/k8s/agentgateway">
            <Translate id="homepage.useCases.integration.gateway">Gateway API</Translate>
          </Link>
          <Link className={styles.integrationChip} to="/docs/installation/k8s/operator">
            <Translate id="homepage.useCases.integration.operator">K8s Operator</Translate>
          </Link>
          <Link className={styles.integrationChip} to="/docs/installation">
            <Translate id="homepage.useCases.integration.local">Local vllm-sr</Translate>
          </Link>
        </div>
        <div className={styles.panelActions}>
          <PillLink to="/docs/intro">
            <Translate id="homepage.useCases.primaryCta">Get hands on</Translate>
          </PillLink>
          <PillLink href="https://github.com/vllm-project/semantic-router" muted rel="noreferrer" target="_blank">
            <Translate id="homepage.useCases.secondaryCta">View on GitHub</Translate>
          </PillLink>
        </div>
      </div>
    </div>
  )
}

export default function UseCaseExplorer(): JSX.Element {
  const [activeId, setActiveId] = useState(useCaseTabs[0].id)

  return (
    <section className={shared.bandSection} aria-labelledby="use-case-explorer-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <ScrollReveal>
          <header className={`${shared.sectionHeader} ${shared.sectionHeaderWide}`}>
            <span className={shared.eyebrow}>
              <Translate id="homepage.useCases.label">How it works</Translate>
            </span>
            <h2 id="use-case-explorer-title" className={shared.sectionTitle}>
              <Translate id="homepage.useCases.title">One router, three use cases</Translate>
            </h2>
            <p className={shared.sectionSubtitle}>
              <Translate id="homepage.useCases.subtitle">See how signals, policies, and model pools connect inside a recipe.</Translate>
            </p>
          </header>
        </ScrollReveal>

        <ScrollReveal delay={80}>
          <div className={`${shared.darkCard} ${styles.shell}`}>
            <div className={styles.tabRow} role="tablist" aria-label="Router use cases">
              {useCaseTabs.map((tab) => {
                const selected = tab.id === activeId
                return (
                  <button
                    key={tab.id}
                    type="button"
                    role="tab"
                    aria-selected={selected}
                    aria-controls={`use-case-panel-${tab.id}`}
                    id={`use-case-tab-${tab.id}`}
                    className={selected ? styles.tabActive : styles.tab}
                    onClick={() => setActiveId(tab.id)}
                  >
                    <span className={styles.tabIndex}>
                      {String(useCaseTabs.indexOf(tab) + 1).padStart(2, '0')}
                    </span>
                    <span className={styles.tabLabel}>{tab.label}</span>
                    <span className={styles.tabSubtitle}>{tab.subtitle}</span>
                  </button>
                )
              })}
            </div>

            <div className={styles.panelStack}>
              {useCaseTabs.map(tab => (
                <UseCasePanel key={tab.id} tab={tab} active={tab.id === activeId} />
              ))}
            </div>
          </div>
        </ScrollReveal>
      </div>
    </section>
  )
}
