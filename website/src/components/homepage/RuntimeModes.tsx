import React from 'react'
import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import { FiArrowRight, FiCode, FiGitBranch } from 'react-icons/fi'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import { SectionLabel } from '@site/src/components/site/Chrome'
import RuntimeArchitecture from './RuntimeArchitecture'
import shared from './homepageShared.module.css'
import styles from './RuntimeModes.module.css'

export default function RuntimeModes(): React.JSX.Element {
  return (
    <section id="runtime-modes" className={styles.compactSection} aria-labelledby="runtime-modes-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <ScrollReveal>
          <header className={`site-section-intro ${shared.sectionHeader} ${styles.compactHeader}`}>
            <SectionLabel>
              <Translate id="homepage.runtimeModes.label">Where it fits</Translate>
            </SectionLabel>
            <h2 id="runtime-modes-title">
              <Translate id="homepage.runtimeModes.title">One frontend. Compose what you need.</Translate>
            </h2>
            <p>
              <Translate id="homepage.runtimeModes.description">
                Route chat through a Decision Engine, or call model intelligence directly. Both paths share the frontend and scalable Serving Engine.
              </Translate>
            </p>
          </header>
        </ScrollReveal>

        <ScrollReveal delay={60}>
          <RuntimeArchitecture />
        </ScrollReveal>

        <div className={styles.modes}>
          <ScrollReveal delay={80}>
            <article className={styles.mode} aria-labelledby="router-mode-title">
              <header className={styles.modeHeader}>
                <span className={styles.modeIcon} aria-hidden="true"><FiGitBranch /></span>
                <div>
                  <span className={styles.modeKicker}>
                    <Translate id="homepage.runtimeModes.router.kicker">Add routing policy</Translate>
                  </span>
                  <h3 id="router-mode-title">
                    <Translate id="homepage.runtimeModes.router.label">Router mode</Translate>
                  </h3>
                </div>
              </header>
              <p className={styles.modeDescription}>
                <Translate id="homepage.runtimeModes.router.description">
                  Route Chat, Messages, and Responses to your model backends. Keep native decisions available alongside routing.
                </Translate>
              </p>
              <pre className={styles.command}><code>vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B</code></pre>
              <p className={styles.modeDetails}>
                <Translate id="homepage.runtimeModes.router.endpoint" values={{ chatEndpoint: <code>/v1/chat/completions</code>, messagesEndpoint: <code>/v1/messages</code> }}>
                  {'Call {chatEndpoint} or {messagesEndpoint}.'}
                </Translate>
              </p>
              <footer className={styles.modeFooter}>
                <Link to="/docs/next/installation">
                  <Translate id="homepage.runtimeModes.router.cta">Run the Router</Translate>
                  <FiArrowRight aria-hidden="true" />
                </Link>
                <Link to="/docs/next/installation/gateway-modes">
                  <Translate id="homepage.runtimeModes.router.gatewayCta">Compare gateway modes</Translate>
                </Link>
              </footer>
            </article>
          </ScrollReveal>

          <ScrollReveal delay={120}>
            <article className={`${styles.mode} ${styles.engineMode}`} aria-labelledby="engine-mode-title">
              <header className={styles.modeHeader}>
                <span className={styles.modeIcon} aria-hidden="true"><FiCode /></span>
                <div>
                  <span className={styles.modeKicker}>
                    <Translate id="homepage.runtimeModes.engine.kicker">Use model intelligence directly</Translate>
                  </span>
                  <h3 id="engine-mode-title">
                    <Translate id="homepage.runtimeModes.engine.label">Engine mode</Translate>
                  </h3>
                </div>
              </header>
              <p className={styles.modeDescription}>
                <Translate id="homepage.runtimeModes.engine.description">
                  Start the same frontend and model workers with routing disabled. Your application owns the next step.
                </Translate>
              </p>
              <pre className={styles.command}><code>vllm-sr serve -e vllm-sr/Decision-2.0-Kai-0.6B</code></pre>
              <p className={styles.modeDetails}>
                <Translate id="homepage.runtimeModes.engine.endpoint" values={{ endpoint: <code>/v1/systemone</code> }}>
                  {'Call {endpoint} with an explicit published model.'}
                </Translate>
              </p>
              <footer className={styles.modeFooter}>
                <Link to="/docs/next/model-runtime/quickstart">
                  <Translate id="homepage.runtimeModes.engine.cta">Run the Engine</Translate>
                  <FiArrowRight aria-hidden="true" />
                </Link>
                <Link to="/docs/next/model-runtime/overview">
                  <Translate id="homepage.runtimeModes.engine.modelsCta">Explore the model runtime</Translate>
                </Link>
              </footer>
            </article>
          </ScrollReveal>
        </div>
        <p className={styles.startupNote}>
          <Link to="/docs/next/installation">
            <Translate id="homepage.runtimeModes.devChannel">Development channel</Translate>
          </Link>
          <span aria-hidden="true"> · </span>
          <Translate id="homepage.runtimeModes.startupNote">Choose the composition at CLI startup.</Translate>
        </p>
      </div>
    </section>
  )
}
