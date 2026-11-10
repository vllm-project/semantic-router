import React, { useState } from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import ThemedImage from '@theme/ThemedImage'
import { FiArrowDown, FiArrowRight, FiCloud, FiCpu, FiGitBranch, FiLayers, FiMessageSquare, FiServer, FiShield, FiZap } from 'react-icons/fi'
import styles from './RuntimeArchitecture.module.css'

type RequestPath = 'chat' | 'native'

function ChatBackends(): React.JSX.Element {
  return (
    <>
      <div className={styles.backendConnector} aria-hidden="true"><FiArrowDown /></div>
      <div className={styles.backends}>
        <FiCloud aria-hidden="true" />
        <div>
          <strong><Translate id="homepage.runtimeModes.diagram.backends">Chat backends</Translate></strong>
          <span><Translate id="homepage.runtimeModes.diagram.generation">Local, private, or cloud generation</Translate></span>
        </div>
      </div>
    </>
  )
}

export default function RuntimeArchitecture(): React.JSX.Element {
  const [path, setPath] = useState<RequestPath>('chat')
  const lightLogo = useBaseUrl('/img/vllm-sr-logo.light.png')
  const darkLogo = useBaseUrl('/img/vllm-sr-logo.white.png')

  return (
    <figure className={styles.architecture} data-path={path} aria-labelledby="runtime-architecture-caption">
      <div className={styles.toolbar}>
        <ThemedImage className={styles.logo} sources={{ light: lightLogo, dark: darkLogo }} alt="vLLM Semantic Router" />
        <div className={styles.pathPicker} role="group" aria-label={translate({ id: 'homepage.runtimeModes.diagram.explore', message: 'Explore a request path' })}>
          <button type="button" aria-pressed={path === 'chat'} onClick={() => setPath('chat')}>
            <FiMessageSquare aria-hidden="true" />
            <Translate id="homepage.runtimeModes.diagram.chat">Routed chat</Translate>
          </button>
          <button type="button" aria-pressed={path === 'native'} onClick={() => setPath('native')}>
            <FiZap aria-hidden="true" />
            <Translate id="homepage.runtimeModes.diagram.native">Native decisions</Translate>
          </button>
        </div>
      </div>

      <div className={styles.diagram}>
        <div className={styles.nativeLane}>
          <span>
            <FiZap aria-hidden="true" />
            <Translate id="homepage.runtimeModes.diagram.direct">Direct native path · bypasses routing</Translate>
          </span>
          <FiArrowDown className={styles.nativeArrow} aria-hidden="true" />
        </div>

        <div className={`${styles.module} ${styles.frontend}`}>
          <div className={styles.moduleHeading}>
            <span className={styles.moduleIcon}><FiLayers aria-hidden="true" /></span>
            <div>
              <span className={styles.eyebrow}><Translate id="homepage.runtimeModes.diagram.shared">Shared entry</Translate></span>
              <h3><Translate id="homepage.runtimeModes.diagram.frontend">Frontend</Translate></h3>
            </div>
          </div>
          <p><Translate id="homepage.runtimeModes.diagram.frontendDescription">Listeners, authentication, and API contracts.</Translate></p>
          <div className={`${styles.api} ${styles.chatApi}`}>
            <span>
              <FiMessageSquare aria-hidden="true" />
              <Translate id="homepage.runtimeModes.diagram.chatApi">Chat & Responses</Translate>
            </span>
            <code>/v1/chat/completions</code>
            <code>/v1/responses</code>
          </div>
          <div className={`${styles.api} ${styles.nativeApi}`}>
            <span>
              <FiZap aria-hidden="true" />
              <Translate id="homepage.runtimeModes.diagram.nativeApi">System One · native decisions</Translate>
            </span>
            <code>/v1/systemone</code>
            <small><Translate id="homepage.runtimeModes.diagram.alias">Alias: /v1/decisions</Translate></small>
          </div>
        </div>

        <div className={`${styles.connector} ${styles.chatConnector}`} aria-hidden="true"><FiArrowRight /></div>

        <div className={`${styles.module} ${styles.decision}`}>
          <div className={styles.moduleHeading}>
            <span className={styles.moduleIcon}><FiGitBranch aria-hidden="true" /></span>
            <div>
              <span className={styles.eyebrow}><Translate id="homepage.runtimeModes.diagram.optional">Optional routing layer</Translate></span>
              <h3><Translate id="homepage.runtimeModes.diagram.decision">Decision Engine</Translate></h3>
            </div>
          </div>
          <p><Translate id="homepage.runtimeModes.diagram.recipe">Each entrypoint resolves to an isolated recipe.</Translate></p>
          <ol className={styles.pipeline}>
            <li>
              <span>01</span>
              <Translate id="homepage.runtimeModes.diagram.signals">Signals</Translate>
            </li>
            <li>
              <span>02</span>
              <Translate id="homepage.runtimeModes.diagram.projections">Projections</Translate>
            </li>
            <li>
              <span>03</span>
              <Translate id="homepage.runtimeModes.diagram.decisions">Decisions</Translate>
            </li>
            <li>
              <span>04</span>
              <Translate id="homepage.runtimeModes.diagram.algorithms">Algorithms</Translate>
            </li>
          </ol>
          <div className={styles.plugins}>
            <FiShield aria-hidden="true" />
            <span><Translate id="homepage.runtimeModes.diagram.plugins">Recipe plugins · request & response hooks</Translate></span>
          </div>
          <div className={styles.mobileBackends}><ChatBackends /></div>
        </div>

        <div className={`${styles.connector} ${styles.taskConnector}`}>
          <span aria-hidden="true">⇄</span>
          <small><Translate id="homepage.runtimeModes.diagram.taskLink">Model tasks</Translate></small>
        </div>

        <div className={`${styles.module} ${styles.serving}`}>
          <div className={styles.moduleHeading}>
            <span className={styles.moduleIcon}><FiCpu aria-hidden="true" /></span>
            <div>
              <span className={styles.eyebrow}><Translate id="homepage.runtimeModes.diagram.execution">Model execution</Translate></span>
              <h3><Translate id="homepage.runtimeModes.diagram.serving">Serving Engine</Translate></h3>
            </div>
          </div>
          <p><Translate id="homepage.runtimeModes.diagram.servingDescription">Choose a model deployment, then a ready replica.</Translate></p>
          <div className={styles.workerPool}>
            <div className={styles.deploymentLabel}>
              <FiServer aria-hidden="true" />
              <Translate id="homepage.runtimeModes.diagram.deployment">Model deployment</Translate>
            </div>
            <div className={styles.replicas}>
              <span>
                <i aria-hidden="true" />
                <Translate id="homepage.runtimeModes.diagram.replicaOne">Replica 1</Translate>
              </span>
              <span>
                <i aria-hidden="true" />
                <Translate id="homepage.runtimeModes.diagram.replicaTwo">Replica 2</Translate>
              </span>
              <span className={styles.moreReplicas}><Translate id="homepage.runtimeModes.diagram.more">+ replicas</Translate></span>
            </div>
            <span className={styles.poolNote}><Translate id="homepage.runtimeModes.diagram.scale">Scale each model independently</Translate></span>
          </div>
          <div className={styles.taskTypes}>
            <span>Choice</span>
            <span>Noul</span>
            <span>Score</span>
            <span>Set</span>
            <span>Span</span>
          </div>
          <span className={styles.workerNote}><Translate id="homepage.runtimeModes.diagram.specialists">Specialist workers: classify · embed · rerank</Translate></span>
        </div>

        <div className={styles.frontendNote}>
          <FiLayers aria-hidden="true" />
          <span><Translate id="homepage.runtimeModes.diagram.logical">Composable capabilities; not fixed process boundaries.</Translate></span>
        </div>

        <div className={styles.backendBranch}><ChatBackends /></div>

        <div className={styles.servingNote}>
          <span className={styles.hardware}>CPU / GPU</span>
          <span><Translate id="homepage.runtimeModes.diagram.ownership">Managed or attached workers</Translate></span>
          <small><Translate id="homepage.runtimeModes.diagram.workerApis">Worker APIs are separate from the public frontend.</Translate></small>
        </div>
      </div>

      <figcaption id="runtime-architecture-caption" className={styles.caption} aria-live="polite" aria-atomic="true">
        <span className={styles.captionDot} aria-hidden="true" />
        {path === 'chat'
          ? <Translate id="homepage.runtimeModes.diagram.chatCaption">Chat follows recipe policy to a generation backend. The Decision Engine calls model workers when signals or plugins need them.</Translate>
          : <Translate id="homepage.runtimeModes.diagram.nativeCaption">Native decisions go from the frontend to an explicit model and ready replica, without the routing pipeline. Available in both Router and Engine modes.</Translate>}
      </figcaption>
    </figure>
  )
}
