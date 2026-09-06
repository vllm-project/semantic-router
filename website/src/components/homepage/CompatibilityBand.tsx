import React from 'react'
import Link from '@docusaurus/Link'
import Translate, { translate } from '@docusaurus/Translate'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import shared from './homepageShared.module.css'
import styles from './CompatibilityBand.module.css'

const deploymentTargets = [
  {
    label: translate({ id: 'homepage.compat.extproc', message: 'Envoy ExtProc' }),
    detail: translate({ id: 'homepage.compat.extproc.detail', message: 'Production gateway filter' }),
    to: '/docs/installation',
  },
  {
    label: translate({ id: 'homepage.compat.gateway', message: 'Gateway API' }),
    detail: translate({ id: 'homepage.compat.gateway.detail', message: 'Agentgateway integration' }),
    to: '/docs/installation/k8s/agentgateway',
  },
  {
    label: translate({ id: 'homepage.compat.operator', message: 'K8s Operator' }),
    detail: translate({ id: 'homepage.compat.operator.detail', message: 'Declarative fleet routing' }),
    to: '/docs/installation/k8s/operator',
  },
  {
    label: translate({ id: 'homepage.compat.local', message: 'Local vllm-sr' }),
    detail: translate({ id: 'homepage.compat.local.detail', message: 'Dev and laptop workflows' }),
    to: '/docs/installation',
  },
]

const signalFamilies = [
  translate({ id: 'homepage.compat.signal.domain', message: 'Domain' }),
  translate({ id: 'homepage.compat.signal.pii', message: 'PII' }),
  translate({ id: 'homepage.compat.signal.jailbreak', message: 'Jailbreak' }),
  translate({ id: 'homepage.compat.signal.preference', message: 'Preference' }),
  translate({ id: 'homepage.compat.signal.embedding', message: 'Embedding' }),
  translate({ id: 'homepage.compat.signal.complexity', message: 'Complexity' }),
  translate({ id: 'homepage.compat.signal.history', message: 'History' }),
  translate({ id: 'homepage.compat.signal.tool', message: 'Tool use' }),
]

export default function CompatibilityBand(): JSX.Element {
  return (
    <section className={shared.bandSection} aria-labelledby="compatibility-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <ScrollReveal>
          <header className={`site-section-intro ${shared.sectionHeader}`}>
            <SectionLabel>
              <Translate id="homepage.compat.label">Deployment options</Translate>
            </SectionLabel>
            <h2 id="compatibility-title" className={shared.sectionTitle}>
              <Translate id="homepage.compat.title">One router, supported deployment paths</Translate>
            </h2>
            <p className={shared.sectionSubtitle}>
              <Translate id="homepage.compat.subtitle">Run behind Envoy, with supported Kubernetes integrations, or locally — and compose up to 20 signal families in a recipe.</Translate>
            </p>
          </header>
        </ScrollReveal>

        <div className={styles.columns}>
          <ScrollReveal delay={60}>
            <div className={`${shared.surfaceCard} ${styles.column}`}>
              <div className={shared.surfaceCardBody}>
                <div className={styles.columnHead}>
                  <h3>
                    <Translate id="homepage.compat.deploy.title">Deployment</Translate>
                  </h3>
                  <p>
                    <Translate id="homepage.compat.deploy.subtitle">
                      Choose a supported integration
                    </Translate>
                  </p>
                </div>
                <ul className={styles.deployList}>
                  {deploymentTargets.map(target => (
                    <li key={target.label}>
                      <Link className={styles.deployCard} to={target.to}>
                        <div className={styles.deployCopy}>
                          <strong>{target.label}</strong>
                          <span>{target.detail}</span>
                        </div>
                        <span className={styles.deployArrow} aria-hidden="true">→</span>
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
              <div className={`${shared.surfaceCardFooter} ${styles.columnCta}`}>
                <PillLink to="/docs/installation" muted>
                  <Translate id="homepage.compat.deploy.cta">View install paths</Translate>
                </PillLink>
              </div>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={120}>
            <div className={`${shared.surfaceCard} ${styles.column}`}>
              <div className={shared.surfaceCardBody}>
                <div className={styles.columnHead}>
                  <h3>
                    <Translate id="homepage.compat.signals.title">Signal families</Translate>
                  </h3>
                  <p>
                    <Translate id="homepage.compat.signals.subtitle">
                      20 request, safety, intent, preference, and system signals
                    </Translate>
                  </p>
                </div>
                <div className={styles.signalPanel}>
                  <div className={styles.signalGrid}>
                    {signalFamilies.map(signal => (
                      <Link
                        key={signal}
                        className={styles.signalChip}
                        to="/docs/tutorials/signal/overview"
                      >
                        {signal}
                      </Link>
                    ))}
                    <span className={styles.signalChipMore} title="20 signal families total">
                      +12 more
                    </span>
                  </div>
                </div>
              </div>
              <div className={`${shared.surfaceCardFooter} ${styles.columnCta}`}>
                <PillLink to="/docs/tutorials/signal/overview">
                  <Translate id="homepage.compat.signals.cta">Explore signals</Translate>
                </PillLink>
              </div>
            </div>
          </ScrollReveal>
        </div>
      </div>
    </section>
  )
}
