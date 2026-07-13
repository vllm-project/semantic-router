import React, { useState } from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import shared from './homepageShared.module.css'
import styles from './WhyDifferentTabs.module.css'

type WhyTabLink = {
  label: string
  to: string
}

type WhyTab = {
  id: string
  label: string
  problem: string
  solutionTitle: string
  solutionBody: string
  links: WhyTabLink[]
}

const whyTabs: WhyTab[] = [
  {
    id: 'security',
    label: translate({
      id: 'homepage.why.security.label',
      message: 'Security & identity',
    }),
    problem: translate({
      id: 'homepage.why.security.problem',
      message:
        'Agents act on behalf of users, invoke tools dynamically, and chain across services. API-level auth was not built for semantic routing decisions.',
    }),
    solutionTitle: translate({
      id: 'homepage.why.security.solutionTitle',
      message: 'Purpose-built for policy-aware routing',
    }),
    solutionBody: translate({
      id: 'homepage.why.security.solutionBody',
      message:
        'PII and jailbreak signals, authz-aware decisions, and per-route plugin policies enforce guardrails before requests reach model backends.',
    }),
    links: [
      { label: 'PII signal', to: '/docs/tutorials/signal/learned/pii' },
      { label: 'Jailbreak signal', to: '/docs/tutorials/signal/learned/jailbreak' },
      { label: 'Authz signal', to: '/docs/tutorials/signal/heuristic/authz' },
    ],
  },
  {
    id: 'observability',
    label: translate({
      id: 'homepage.why.observability.label',
      message: 'Observability',
    }),
    problem: translate({
      id: 'homepage.why.observability.problem',
      message:
        'Debugging routing requires visibility into signals, model selection, token usage, and session context — not just HTTP access logs.',
    }),
    solutionTitle: translate({
      id: 'homepage.why.observability.solutionTitle',
      message: 'Full-stack routing observability',
    }),
    solutionBody: translate({
      id: 'homepage.why.observability.solutionBody',
      message:
        'Router replay captures decision metadata, projection traces, and cost fields. Dashboard insights aggregate decisions, models, and signals in one place.',
    }),
    links: [
      { label: 'Router replay', to: '/docs/tutorials/plugin/router-replay' },
      { label: 'API & observability', to: '/docs/tutorials/global/api-and-observability' },
      { label: 'Dashboard', to: '/docs/installation' },
    ],
  },
  {
    id: 'performance',
    label: translate({
      id: 'homepage.why.performance.label',
      message: 'Performance & scale',
    }),
    problem: translate({
      id: 'homepage.why.performance.problem',
      message:
        'Every request runs multiple signal evaluations and plugin hooks. Gateway overhead compounds across agent tool chains and high-QPS fleets.',
    }),
    solutionTitle: translate({
      id: 'homepage.why.performance.solutionTitle',
      message: 'Signal layer optimized for inference paths',
    }),
    solutionBody: translate({
      id: 'homepage.why.performance.solutionBody',
      message:
        'Rust bindings for classification and embeddings, semantic caching, and async replay writes keep routing overhead predictable at scale.',
    }),
    links: [
      { label: 'Semantic cache', to: '/docs/tutorials/plugin/semantic-cache' },
      { label: 'Signal overview', to: '/docs/tutorials/signal/overview' },
      { label: 'Installation', to: '/docs/installation' },
    ],
  },
  {
    id: 'cost',
    label: translate({
      id: 'homepage.why.cost.label',
      message: 'Cost optimization',
    }),
    problem: translate({
      id: 'homepage.why.cost.problem',
      message:
        'Agentic workloads spiral cost fast. Static model routing cannot attribute spend, compare baselines, or route cheaper models when quality allows.',
    }),
    solutionTitle: translate({
      id: 'homepage.why.cost.solutionTitle',
      message: 'Routing that understands model economics',
    }),
    solutionBody: translate({
      id: 'homepage.why.cost.solutionBody',
      message:
        'Per-model pricing, baseline cost comparison, and multi_factor tradeoffs let operators optimize quality, latency, and spend per decision.',
    }),
    links: [
      { label: 'Multi-factor routing', to: '/docs/tutorials/algorithm/selection/multi-factor' },
      { label: 'Router replay cost fields', to: '/docs/tutorials/global/api-and-observability' },
      { label: 'Model configuration', to: '/docs/installation/configuration' },
    ],
  },
]

export default function WhyDifferentTabs(): JSX.Element {
  const [activeId, setActiveId] = useState(whyTabs[0].id)
  const activeTab = whyTabs.find(tab => tab.id === activeId) ?? whyTabs[0]

  return (
    <section className={shared.darkSection} aria-labelledby="why-different-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <header className={shared.sectionHeader}>
          <SectionLabel className={styles.sectionLabel}>
            <Translate id="homepage.why.label">Why now</Translate>
          </SectionLabel>
          <h2 id="why-different-title" className={shared.sectionTitle}>
            <Translate id="homepage.why.title">
              LLM routing is not a reverse proxy problem
            </Translate>
          </h2>
          <p className={shared.sectionSubtitle}>
            <Translate id="homepage.why.subtitle">
              Retrofitting generic gateways for semantic signals creates blind spots.
              See how purpose-built routing addresses security, observability,
              performance, and cost.
            </Translate>
          </p>
        </header>

        <div className={styles.tabRow} role="tablist">
          {whyTabs.map(tab => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={tab.id === activeId}
              className={tab.id === activeId ? styles.tabActive : styles.tab}
              onClick={() => setActiveId(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className={styles.panel} role="tabpanel">
          <div className={styles.problemBlock}>
            <span className={styles.problemLabel}>
              <Translate id="homepage.why.problemLabel">Problem</Translate>
            </span>
            <p>{activeTab.problem}</p>
          </div>

          <div className={styles.solutionBlock}>
            <span className={styles.solutionLabel}>vLLM Semantic Router</span>
            <h3>{activeTab.solutionTitle}</h3>
            <p>{activeTab.solutionBody}</p>
            <div className={styles.linkRow}>
              {activeTab.links.map(link => (
                <PillLink key={link.to} to={link.to} muted>
                  {link.label}
                </PillLink>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
