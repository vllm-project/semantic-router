import React from 'react'
import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import ThemedImage from '@theme/ThemedImage'
import { FiCheck, FiCloud, FiCode, FiCpu, FiLayers, FiMessageSquare, FiServer } from 'react-icons/fi'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import { SectionLabel } from '@site/src/components/site/Chrome'
import shared from './homepageShared.module.css'
import styles from './RuntimeModes.module.css'

function RuntimeLogo({ mark = false }: { mark?: boolean }): React.JSX.Element {
  const lightLogo = useBaseUrl('/img/vllm-sr-logo.light.png')
  const darkLogo = useBaseUrl('/img/vllm-sr-logo.white.png')
  const sources = {
    light: lightLogo,
    dark: mark ? lightLogo : darkLogo,
  }

  return (
    <span className={mark ? styles.brandMark : styles.brandLogo}>
      <ThemedImage sources={sources} alt="vLLM Semantic Router" />
    </span>
  )
}

export default function RuntimeModes(): React.JSX.Element {
  return (
    <section id="runtime-modes" className={shared.bandSection} aria-labelledby="runtime-modes-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <ScrollReveal>
          <header className={`site-section-intro ${shared.sectionHeader}`}>
            <SectionLabel>
              <Translate id="homepage.runtimeModes.label">Where it fits</Translate>
            </SectionLabel>
            <h2 id="runtime-modes-title">
              <Translate id="homepage.runtimeModes.title">Route inference. Build with intelligence.</Translate>
            </h2>
            <p>
              <Translate id="homepage.runtimeModes.description">
                Run the Router for model orchestration, or call its model runtime directly from your application.
              </Translate>
            </p>
          </header>
        </ScrollReveal>

        <div className={styles.modes}>
          <ScrollReveal delay={60}>
            <article className={styles.mode} aria-labelledby="router-mode-title">
              <header className={styles.modeHeader}>
                <div className={styles.modeIdentity}>
                  <span className={styles.modeIcon} aria-hidden="true"><RuntimeLogo mark /></span>
                  <div>
                    <span className={styles.modeKicker}>
                      <Translate id="homepage.runtimeModes.router.kicker">Orchestrate inference</Translate>
                    </span>
                    <h3 id="router-mode-title">
                      <Translate id="homepage.runtimeModes.router.label">Router mode</Translate>
                    </h3>
                  </div>
                  <span className={styles.modeIndex} aria-hidden="true">01</span>
                </div>
                <p className={styles.modeStatement}>
                  <Translate id="homepage.runtimeModes.router.title">One endpoint for your model fleet.</Translate>
                </p>
                <p className={styles.modeDescription}>
                  <Translate id="homepage.runtimeModes.router.description">
                    Select or combine model backends behind one OpenAI-compatible API.
                  </Translate>
                </p>
              </header>

              <div className={styles.modeVisual}>
                <div className={styles.visualClient}>
                  <FiMessageSquare aria-hidden="true" />
                  <span><Translate id="homepage.runtimeModes.router.flow.client">Apps & agents</Translate></span>
                </div>
                <span className={styles.connector} aria-hidden="true" />
                <div className={styles.visualCore}>
                  <RuntimeLogo />
                  <strong><Translate id="homepage.runtimeModes.router.coreLabel">Router</Translate></strong>
                </div>
                <span className={styles.connector} aria-hidden="true" />
                <div className={styles.backends}>
                  <div>
                    <FiCloud aria-hidden="true" />
                    <span><Translate id="homepage.runtimeModes.router.backend.cloud">Cloud</Translate></span>
                  </div>
                  <div className={styles.selectedBackend}>
                    <FiServer aria-hidden="true" />
                    <span><Translate id="homepage.runtimeModes.router.backend.private">Private</Translate></span>
                    <FiCheck aria-hidden="true" />
                  </div>
                  <div>
                    <FiCpu aria-hidden="true" />
                    <span><Translate id="homepage.runtimeModes.router.backend.local">Local</Translate></span>
                  </div>
                </div>
                <div className={styles.visualCaption}>
                  <FiLayers aria-hidden="true" />
                  <Translate id="homepage.runtimeModes.router.visualCaption">Select, cascade, or combine models</Translate>
                </div>
              </div>

              <div className={styles.commandBlock}>
                <Link className={styles.channelBadge} to="/docs/installation">
                  <Translate id="homepage.runtimeModes.devChannel">Development channel</Translate>
                </Link>
                <pre className={styles.command}><code>vllm-sr serve --gateway standalone</code></pre>
              </div>

              <div className={styles.modeDetails}>
                <p>
                  <Translate
                    id="homepage.runtimeModes.router.extproc"
                    values={{ flag: <code>--gateway extproc</code> }}
                  >
                    {'Native HTTP gateway. Add Envoy with {flag}.'}
                  </Translate>
                </p>
              </div>

              <footer className={styles.modeFooter}>
                <Link to="/docs/installation">
                  <Translate id="homepage.runtimeModes.router.cta">Run the Router</Translate>
                  <span aria-hidden="true"> →</span>
                </Link>
                <Link to="/docs/installation/gateway-modes">
                  <Translate id="homepage.runtimeModes.router.gatewayCta">Compare gateway modes</Translate>
                </Link>
              </footer>
            </article>
          </ScrollReveal>

          <ScrollReveal delay={100}>
            <article className={`${styles.mode} ${styles.engineMode}`} aria-labelledby="engine-mode-title">
              <header className={styles.modeHeader}>
                <div className={styles.modeIdentity}>
                  <span className={styles.modeIcon} aria-hidden="true"><RuntimeLogo mark /></span>
                  <div>
                    <span className={styles.modeKicker}>
                      <Translate id="homepage.runtimeModes.engine.kicker">Run task models</Translate>
                    </span>
                    <h3 id="engine-mode-title">
                      <Translate id="homepage.runtimeModes.engine.label">Engine mode</Translate>
                    </h3>
                  </div>
                  <span className={styles.modeIndex} aria-hidden="true">02</span>
                </div>
                <p className={styles.modeStatement}>
                  <Translate id="homepage.runtimeModes.engine.title">Model intelligence for your own code.</Translate>
                </p>
                <p className={styles.modeDescription}>
                  <Translate id="homepage.runtimeModes.engine.description">
                    Serve decision models, classifiers, embeddings, and rerankers for your own application.
                  </Translate>
                </p>
              </header>

              <div className={styles.modeVisual}>
                <div className={styles.visualClient}>
                  <FiCode aria-hidden="true" />
                  <span><Translate id="homepage.runtimeModes.engine.flow.client">Your application</Translate></span>
                </div>
                <span className={styles.connector} aria-hidden="true" />
                <div className={`${styles.visualCore} ${styles.engineCore}`}>
                  <RuntimeLogo />
                  <strong><Translate id="homepage.runtimeModes.engine.coreLabel">Engine</Translate></strong>
                </div>
                <span className={styles.connector} aria-hidden="true" />
                <div className={styles.results}>
                  <span className={styles.resultType}><Translate id="homepage.runtimeModes.engine.typedResults">Typed results</Translate></span>
                  <dl className={styles.answerTypes}>
                    <div>
                      <dt>Choice</dt>
                      <dd><code>"code"</code></dd>
                    </div>
                    <div>
                      <dt>Noul</dt>
                      <dd>
                        <code>0.94</code>
                        <span>P(true)</span>
                      </dd>
                    </div>
                    <div>
                      <dt>Score</dt>
                      <dd><code>4.2</code></dd>
                    </div>
                    <div>
                      <dt>Set</dt>
                      <dd><code>["pii"]</code></dd>
                    </div>
                    <div>
                      <dt>Span</dt>
                      <dd><code>start:end</code></dd>
                    </div>
                  </dl>
                  <span className={styles.resultType}><Translate id="homepage.runtimeModes.engine.vectors">Vectors & rankings</Translate></span>
                </div>
                <div className={styles.visualCaption}>
                  <FiCode aria-hidden="true" />
                  <Translate id="homepage.runtimeModes.engine.visualCaption">Model outputs. Your application logic.</Translate>
                </div>
              </div>

              <div className={styles.commandBlock}>
                <Link className={styles.channelBadge} to="/docs/installation">
                  <Translate id="homepage.runtimeModes.devChannel">Development channel</Translate>
                </Link>
                <pre className={styles.command}><code>vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100</code></pre>
              </div>

              <div className={styles.modeDetails}>
                <p>
                  <Translate id="homepage.runtimeModes.engine.tasks">
                    Model-dependent APIs.
                  </Translate>
                </p>
                <ul className={styles.endpoints}>
                  {['/v1/decisions', '/v1/classify', '/v1/embeddings', '/v1/rerank'].map(endpoint => (
                    <li key={endpoint}><code>{endpoint}</code></li>
                  ))}
                </ul>
              </div>

              <footer className={styles.modeFooter}>
                <Link to="/docs/model-runtime/quickstart">
                  <Translate id="homepage.runtimeModes.engine.cta">Run the Engine</Translate>
                  <span aria-hidden="true"> →</span>
                </Link>
                <Link to="/docs/model-runtime/overview">
                  <Translate id="homepage.runtimeModes.engine.modelsCta">Explore the model runtime</Translate>
                </Link>
              </footer>
            </article>
          </ScrollReveal>
        </div>

      </div>
    </section>
  )
}
