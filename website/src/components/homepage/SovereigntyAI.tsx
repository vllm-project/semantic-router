import React, { useEffect, useRef, useState } from 'react'
import Translate from '@docusaurus/Translate'
import Claude from '@lobehub/icons/es/Claude/components/Mono'
import DeepSeek from '@lobehub/icons/es/DeepSeek/components/Mono'
import Gemini from '@lobehub/icons/es/Gemini/components/Mono'
import Meta from '@lobehub/icons/es/Meta/components/Mono'
import OpenAI from '@lobehub/icons/es/OpenAI/components/Mono'
import Qwen from '@lobehub/icons/es/Qwen/components/Mono'
import { FiCheck, FiCloud, FiCpu, FiDatabase, FiGlobe, FiLock, FiShield } from 'react-icons/fi'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import SovereigntyPolicyGate from './SovereigntyPolicyGate'
import shared from './homepageShared.module.css'
import styles from './SovereigntyAI.module.css'

const modelGroups = {
  open: [
    { name: 'DeepSeek', Icon: DeepSeek },
    { name: 'Qwen', Icon: Qwen },
    { name: 'Llama', Icon: Meta },
  ],
  closed: [
    { name: 'Claude', Icon: Claude },
    { name: 'GPT', Icon: OpenAI },
    { name: 'Gemini', Icon: Gemini },
  ],
}

function ModelFleet({ group }: { group: keyof typeof modelGroups }): React.JSX.Element {
  return (
    <div className={styles.modelFleet}>
      {modelGroups[group].map(({ name, Icon }) => (
        <span key={name} className={styles.modelBadge}>
          <Icon size={28} aria-hidden="true" />
          <span>{name}</span>
        </span>
      ))}
    </div>
  )
}

export default function SovereigntyAI(): React.JSX.Element {
  const [externalRequest, setExternalRequest] = useState(false)
  const visualRef = useRef<HTMLElement>(null)

  useEffect(() => {
    const visual = visualRef.current
    if (!visual) return

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
    let visible = false
    let timer: number | undefined

    const synchronize = (): void => {
      window.clearInterval(timer)
      timer = undefined
      if (!visible || reducedMotion.matches || document.hidden) {
        setExternalRequest(false)
        return
      }
      timer = window.setInterval(() => setExternalRequest(current => !current), 7000)
    }

    const observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting
      synchronize()
    }, { threshold: 0.25 })

    observer.observe(visual)
    reducedMotion.addEventListener('change', synchronize)
    document.addEventListener('visibilitychange', synchronize)

    return () => {
      window.clearInterval(timer)
      observer.disconnect()
      reducedMotion.removeEventListener('change', synchronize)
      document.removeEventListener('visibilitychange', synchronize)
    }
  }, [])

  return (
    <section id="sovereignty-ai" className={shared.bandSection} aria-labelledby="sovereignty-ai-title">
      <div className={'site-shell-container ' + shared.sectionInner}>
        <div className={styles.layout}>
          <ScrollReveal>
            <div className={styles.copy}>
              <header className={'site-section-intro ' + styles.header}>
                <SectionLabel>
                  <Translate id="homepage.sovereigntyAI.label">Sovereignty AI</Translate>
                </SectionLabel>
                <h2 id="sovereignty-ai-title">
                  <Translate id="homepage.sovereigntyAI.title">Your knowledge. Your control.</Translate>
                </h2>
                <p>
                  <Translate id="homepage.sovereigntyAI.description">
                    Keep private domain knowledge on premises with open models running on your GPUs. Use external frontier models for requests your routing policy allows.
                  </Translate>
                </p>
              </header>

              <ul className={styles.principles}>
                <li>
                  <FiShield aria-hidden="true" />
                  <span><Translate id="homepage.sovereigntyAI.privacy">Privacy-aware request routing</Translate></span>
                </li>
                <li>
                  <FiCpu aria-hidden="true" />
                  <span><Translate id="homepage.sovereigntyAI.localCompute">On-prem GPUs. Open models.</Translate></span>
                </li>
                <li>
                  <FiCloud aria-hidden="true" />
                  <span><Translate id="homepage.sovereigntyAI.frontier">Frontier models, under your policy.</Translate></span>
                </li>
              </ul>

              <PillLink to="/docs/overview/signal-driven-decisions" muted>
                <Translate id="homepage.sovereigntyAI.cta">Explore routing policies</Translate>
              </PillLink>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={80}>
            <figure
              ref={visualRef}
              className={styles.visual}
              data-route={externalRequest ? 'external' : 'private'}
              aria-labelledby="sovereignty-ai-visual-title"
              aria-describedby="sovereignty-ai-diagram-description"
            >
              <div className={styles.visualHeader}>
                <span id="sovereignty-ai-visual-title" className={styles.visualTitle}>
                  <FiShield aria-hidden="true" />
                  <Translate id="homepage.sovereigntyAI.visualTitle">Privacy-aware routing</Translate>
                </span>
                <span className={styles.policyBadge}>
                  <span aria-hidden="true" />
                  <Translate id="homepage.sovereigntyAI.policyControl">Policy in control</Translate>
                </span>
              </div>

              <div className={styles.request} key={externalRequest ? 'external' : 'private'}>
                <span className={styles.requestIcon}>
                  {externalRequest ? <FiGlobe aria-hidden="true" /> : <FiDatabase aria-hidden="true" />}
                </span>
                <div className={styles.requestCopy}>
                  <span><Translate id="homepage.sovereigntyAI.incomingRequest">Incoming request</Translate></span>
                  <strong>
                    {externalRequest
                      ? <Translate id="homepage.sovereigntyAI.publicRequest">Public research</Translate>
                      : <Translate id="homepage.sovereigntyAI.privateRequest">Internal domain knowledge</Translate>}
                  </strong>
                </div>
                <span className={styles.requestConstraint}>
                  {externalRequest ? <FiCheck aria-hidden="true" /> : <FiLock aria-hidden="true" />}
                  {externalRequest
                    ? <Translate id="homepage.sovereigntyAI.externalEligible">External eligible</Translate>
                    : <Translate id="homepage.sovereigntyAI.keepOnPrem">Keep on-prem</Translate>}
                </span>
              </div>

              <div className={styles.intake} aria-hidden="true"><span /></div>

              <SovereigntyPolicyGate externalRequest={externalRequest} />

              <svg className={styles.routes} viewBox="0 0 600 52" preserveAspectRatio="none" aria-hidden="true">
                <path className={styles.routeLocal} d="M 300 0 V 14 Q 300 24 290 24 H 157 Q 147 24 147 34 V 52" />
                <path className={styles.routeExternal} d="M 300 0 V 14 Q 300 24 310 24 H 443 Q 453 24 453 34 V 52" />
                <path className={styles.localPacket} d="M 300 0 V 14 Q 300 24 290 24 H 157 Q 147 24 147 34 V 52" />
                <path className={styles.externalPacket} d="M 300 0 V 14 Q 300 24 310 24 H 443 Q 453 24 453 34 V 52" />
                <g className={styles.stopMarker}>
                  <circle cx="399" cy="24" r="7" />
                  <path d="M 397 22 L 401 26 M 401 22 L 397 26" />
                </g>
              </svg>

              <div className={styles.destinations}>
                <div className={styles.localDestination}>
                  <div className={styles.destinationLabel}>
                    <FiCpu aria-hidden="true" />
                    <Translate id="homepage.sovereigntyAI.onPremGPU">On-prem GPU</Translate>
                  </div>
                  <strong className={styles.modelTitle}>
                    <Translate id="homepage.sovereigntyAI.openModels">Open models</Translate>
                  </strong>
                  <ModelFleet group="open" />
                  <span className={styles.destinationDetail}>
                    <Translate id="homepage.sovereigntyAI.knowledgeOnPrem">Domain knowledge stays on-prem</Translate>
                  </span>
                  <span className={styles.localStatus}>
                    <FiCheck aria-hidden="true" />
                    {externalRequest
                      ? <Translate id="homepage.sovereigntyAI.localAvailable">Eligible local option</Translate>
                      : <Translate id="homepage.sovereigntyAI.localSelected">Selected for private context</Translate>}
                  </span>
                </div>

                <div className={styles.externalDestination}>
                  <div className={styles.destinationLabel}>
                    <FiGlobe aria-hidden="true" />
                    <Translate id="homepage.sovereigntyAI.externalProvider">External provider</Translate>
                  </div>
                  <strong className={styles.modelTitle}>
                    <Translate id="homepage.sovereigntyAI.closedModels">Frontier closed models</Translate>
                  </strong>
                  <ModelFleet group="closed" />
                  <span className={styles.destinationDetail}>
                    <Translate id="homepage.sovereigntyAI.externalCondition">Only policy-eligible requests</Translate>
                  </span>
                  <span className={styles.externalStatus}>
                    {externalRequest ? <FiCheck aria-hidden="true" /> : <FiLock aria-hidden="true" />}
                    {externalRequest
                      ? <Translate id="homepage.sovereigntyAI.frontierSelected">Selected within policy</Translate>
                      : <Translate id="homepage.sovereigntyAI.privateExcluded">Excluded for private context</Translate>}
                  </span>
                </div>
              </div>

              <figcaption className={styles.caption}>
                <span className={styles.selectionOrder}>
                  <FiShield aria-hidden="true" />
                  <Translate id="homepage.sovereigntyAI.eligibilityFirst">Eligibility first</Translate>
                  <span aria-hidden="true">→</span>
                  <Translate id="homepage.sovereigntyAI.selectionSecond">Model selection</Translate>
                </span>
                <span className={styles.objectives}>
                  <Translate id="homepage.sovereigntyAI.objectives">Quality · latency · cost</Translate>
                </span>
              </figcaption>
              <p id="sovereignty-ai-diagram-description" className={styles.accessibleDescription}>
                <Translate id="homepage.sovereigntyAI.diagramDescription">
                  This animation illustrates a configured privacy policy: internal domain knowledge is routed to open models on on-prem GPUs, while public requests can use external frontier closed models when eligible. Privacy, locality, and access constraints apply before model selection.
                </Translate>
              </p>
            </figure>
          </ScrollReveal>
        </div>
      </div>
    </section>
  )
}
