import React, { useRef } from 'react'
import clsx from 'clsx'
import Link from '@docusaurus/Link'
import Translate, { translate } from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import { SectionLabel } from '@site/src/components/site/Chrome'
import PixelIcon, { type PixelIconName } from '@site/src/components/duo/PixelIcon'
import Terminal from '@site/src/components/duo/Terminal'
import usePauseOffscreen from '@site/src/components/duo/usePauseOffscreen'
import duo from '@site/src/components/duo/duo.module.css'
import styles from './RuntimeModes.module.css'

type Vars = React.CSSProperties & Record<`--${string}`, string>

// Each request takes one third of the loop: a client's block travels to the Router, then on to one model in the pool.
const REQUESTS = [
  { client: 0, row: 1, col: 1 },
  { client: 1, row: 0, col: 3 },
  { client: 2, row: 2, col: 2 },
]
const CLIENT_Y = [20, 84, 148]
const ROW_Y = [60, 84, 108]
const LOOP = 7.2

function delay(index: number): string {
  return `${-((REQUESTS.length - index) % REQUESTS.length) * (LOOP / REQUESTS.length)}s`
}

function Plug({ to, children }: { to: string, children: React.ReactNode }): JSX.Element {
  return (
    <li>
      <Link className={styles.plug} to={to}>{children}</Link>
    </li>
  )
}

function Client({ icon, children }: { icon: PixelIconName, children: React.ReactNode }): JSX.Element {
  return (
    <li className={styles.client}>
      <PixelIcon name={icon} size={16} />
      <span>{children}</span>
    </li>
  )
}

/* Router mode's place: clients on the left, the Router in the middle, your model pool on the right. */
function RouterMap(): JSX.Element {
  const mark = useBaseUrl('/img/vllm-sr-mark.svg')
  return (
    <figure className={styles.map}>
      <figcaption className={styles.mapCap}>
        <Translate id="homepage.runtimeModes.map.router.caption">Between your clients and your model pool</Translate>
        <span className={styles.srOnly}>
          {' '}
          <Translate id="homepage.runtimeModes.map.router.description">
            Apps, agents and IDEs call the Router, which sends each request to one model in your pool.
          </Translate>
        </span>
      </figcaption>
      <div className={styles.mapGrid} aria-hidden="true">
        <ul className={styles.clients}>
          <Client icon="client"><Translate id="homepage.runtimeModes.map.apps">Apps</Translate></Client>
          <Client icon="agent"><Translate id="homepage.runtimeModes.map.agents">Agents</Translate></Client>
          <Client icon="code"><Translate id="homepage.runtimeModes.map.ides">IDEs</Translate></Client>
        </ul>

        <div className={styles.conn}>
          <svg viewBox="0 0 40 168" preserveAspectRatio="none">
            {CLIENT_Y.map(y => <path key={y} d={`M0 ${y}H20V84H40`} vectorEffect="non-scaling-stroke" />)}
          </svg>
          {REQUESTS.map((request, index) => (
            <i
              key={index}
              className={clsx(styles.pkt, styles.fanIn)}
              style={{ '--y': `${CLIENT_Y[request.client]}px`, 'animationDelay': delay(index) } as Vars}
            />
          ))}
        </div>

        <div className={clsx(styles.hub, styles.hubRouter)}>
          <img className={styles.hubLogo} src={mark} alt="" width={22} height={22} />
          <b><Translate id="homepage.runtimeModes.map.router">Router</Translate></b>
        </div>

        <div className={styles.conn}>
          <svg viewBox="0 0 40 168" preserveAspectRatio="none">
            {ROW_Y.map(y => <path key={y} d={`M0 84H20V${y}H40`} vectorEffect="non-scaling-stroke" />)}
          </svg>
          {REQUESTS.map((request, index) => (
            <i
              key={index}
              className={clsx(styles.pkt, styles.fanOut)}
              style={{ '--y': `${ROW_Y[request.row]}px`, 'animationDelay': delay(index) } as Vars}
            />
          ))}
        </div>

        <div className={styles.poolWrap}>
          <div className={styles.pool}>
            {Array.from({ length: 12 }, (_, cell) => {
              const row = Math.floor(cell / 4)
              const col = cell % 4
              const hit = REQUESTS.findIndex(request => request.row === row && request.col === col)
              return (
                <span key={cell} className={clsx(styles.block, (cell * 5) % 7 < 2 && styles.blockSky)}>
                  {hit >= 0 && (
                    <i
                      className={clsx(styles.lit, hit === 0 && styles.litStill)}
                      style={{ animationDelay: delay(hit) }}
                    />
                  )}
                </span>
              )
            })}
          </div>
          <span className={styles.poolCap}>
            <Translate id="homepage.runtimeModes.map.pool">Your model pool</Translate>
          </span>
        </div>
      </div>
    </figure>
  )
}

const ANSWERS = [
  { id: 'engineering', label: <Translate id="homepage.runtimeModes.map.answer.engineering">engineering</Translate>, blocks: 7 },
  { id: 'billing', label: <Translate id="homepage.runtimeModes.map.answer.billing">billing</Translate>, blocks: 2 },
  { id: 'sales', label: <Translate id="homepage.runtimeModes.map.answer.sales">sales</Translate>, blocks: 1 },
]

/* Engine mode's place: your app asks, the Engine on your hardware answers with a probability for every option. */
function EngineMap(): JSX.Element {
  return (
    <figure className={styles.map}>
      <figcaption className={styles.mapCap}>
        <Translate id="homepage.runtimeModes.map.engine.caption">Beside your app, on your own hardware</Translate>
        <span className={styles.srOnly}>
          {' '}
          <Translate id="homepage.runtimeModes.map.engine.description">
            Your app asks the Engine a question and gets back a probability for each option.
          </Translate>
        </span>
      </figcaption>
      <div className={styles.mapGrid} aria-hidden="true">
        <ul className={clsx(styles.clients, styles.single)}>
          <Client icon="agent"><Translate id="homepage.runtimeModes.map.app">Your app</Translate></Client>
        </ul>

        <div className={styles.conn}>
          <svg viewBox="0 0 40 168" preserveAspectRatio="none">
            <path d="M0 84H40" vectorEffect="non-scaling-stroke" />
          </svg>
          <i className={clsx(styles.pkt, styles.ask)} />
        </div>

        <div className={clsx(styles.hub, styles.hubEngine)}>
          <PixelIcon name="bolt" size={22} />
          <b><Translate id="homepage.runtimeModes.map.engine">Engine</Translate></b>
          <span className={styles.chipsRow}>
            {[0, 1, 2, 3].map(index => <i key={index} style={{ animationDelay: `${0.6 + index * 0.08}s` }} />)}
          </span>
          <span className={styles.hubSub}>
            <Translate id="homepage.runtimeModes.map.hardware">GPU or CPU</Translate>
          </span>
        </div>

        <div className={styles.conn}>
          <svg viewBox="0 0 40 168" preserveAspectRatio="none">
            <path d="M0 84H40" vectorEffect="non-scaling-stroke" />
          </svg>
          <i className={clsx(styles.pkt, styles.reply)} />
        </div>

        <ul className={styles.answers}>
          {ANSWERS.map(answer => (
            <li key={answer.id}>
              <span>{answer.label}</span>
              <span className={styles.bar}>
                <i style={{ '--n': String(answer.blocks), 'animationTimingFunction': `steps(${answer.blocks}, end)` } as Vars} />
              </span>
            </li>
          ))}
        </ul>
      </div>
    </figure>
  )
}

/* Where vLLM-SR runs and plugs in: Router mode between your clients and your models, Engine mode beside your app. */
export default function RuntimeModes(): React.JSX.Element {
  const sectionRef = useRef<HTMLElement>(null)
  usePauseOffscreen(sectionRef)

  return (
    <section ref={sectionRef} id="runtime-modes" className={styles.section} aria-labelledby="runtime-modes-title">
      <div className={styles.wrap}>
        <ScrollReveal>
          <header className="site-section-intro">
            <SectionLabel>
              <Translate id="homepage.runtimeModes.label">Where it fits</Translate>
            </SectionLabel>
            <h2 id="runtime-modes-title">
              <Translate id="homepage.runtimeModes.title">In front of your models, or beside your app.</Translate>
            </h2>
            <p>
              <Translate id="homepage.runtimeModes.description">
                One install, two places to plug in, on the stack you already run.
              </Translate>
            </p>
          </header>
        </ScrollReveal>

        <ScrollReveal delay={80}>
          <div className={styles.grid}>
            <article className={clsx(styles.sys, styles.two)} aria-labelledby="router-mode-title">
              <p className={styles.eyebrow}>
                <i className={clsx(duo.sysDot, duo.sysDotSky)} aria-hidden="true" />
                <Translate id="homepage.runtimeModes.router.kicker">Add routing policy</Translate>
              </p>
              <h3 id="router-mode-title" className={styles.title}>
                <Translate id="homepage.runtimeModes.router.label">Router mode</Translate>
              </h3>
              <p className={styles.api}>
                <Translate id="homepage.runtimeModes.router.endpoint" values={{ chatEndpoint: <code>/v1/chat/completions</code>, messagesEndpoint: <code>/v1/messages</code> }}>
                  {'Call {chatEndpoint} or {messagesEndpoint}.'}
                </Translate>
              </p>
              <p className={styles.what}>
                <Translate id="homepage.runtimeModes.router.description">
                  Routes Chat, Messages and Responses to your backends, with native decisions alongside.
                </Translate>
              </p>
              <RouterMap />
              <Terminal
                className={styles.code}
                text="vllm-sr serve vllm-sr/vela3-fast"
                caption={<Translate id="homepage.install.caption.human">terminal</Translate>}
              />
              <ul className={styles.plugs} aria-label={translate({ id: 'homepage.runtimeModes.plug.router.aria', message: 'Router mode integrations' })}>
                <Plug to="/docs/installation/protocol-compatibility">
                  <Translate id="homepage.runtimeModes.plug.openai">OpenAI-compatible API</Translate>
                </Plug>
                <Plug to="/docs/installation/agent-harness">
                  <Translate id="homepage.runtimeModes.plug.harness">Agent harnesses</Translate>
                </Plug>
                <Plug to="/docs/installation/gateway-modes">
                  <Translate id="homepage.runtimeModes.plug.gateways">Envoy gateways</Translate>
                </Plug>
                <Plug to="/docs/installation/k8s/gateways">
                  <Translate id="homepage.runtimeModes.plug.k8s">Kubernetes</Translate>
                </Plug>
              </ul>
              <footer className={styles.foot}>
                <Link className={duo.more} to="/docs/installation">
                  <Translate id="homepage.runtimeModes.router.cta">Run the Router</Translate>
                </Link>
                <Link className={duo.more} to="/docs/installation/gateway-modes">
                  <Translate id="homepage.runtimeModes.router.gatewayCta">Compare gateway modes</Translate>
                </Link>
              </footer>
            </article>

            <article className={styles.sys} aria-labelledby="engine-mode-title">
              <p className={styles.eyebrow}>
                <i className={duo.sysDot} aria-hidden="true" />
                <Translate id="homepage.runtimeModes.engine.kicker">Use model intelligence directly</Translate>
              </p>
              <h3 id="engine-mode-title" className={styles.title}>
                <Translate id="homepage.runtimeModes.engine.label">Engine mode</Translate>
              </h3>
              <p className={styles.api}>
                <Translate id="homepage.runtimeModes.engine.endpoint" values={{ endpoint: <code>/v1/systemone</code> }}>
                  {'Call {endpoint} with an explicit published model.'}
                </Translate>
              </p>
              <p className={styles.what}>
                <Translate id="homepage.runtimeModes.engine.description">
                  The same frontend and workers, routing off. Your application owns the next step.
                </Translate>
              </p>
              <EngineMap />
              <Terminal
                className={styles.code}
                text="vllm-sr serve -e vllm-sr/d3-edge"
                caption={<Translate id="homepage.install.caption.human">terminal</Translate>}
              />
              <ul className={styles.plugs} aria-label={translate({ id: 'homepage.runtimeModes.plug.engine.aria', message: 'Engine mode integrations' })}>
                <Plug to="/docs/system-one/use/api">
                  <Translate id="homepage.runtimeModes.plug.systemone">System One API</Translate>
                </Plug>
                <Plug to="/docs/installation/support-matrix">
                  <Translate id="homepage.runtimeModes.plug.gpus">On-prem GPUs</Translate>
                </Plug>
                <Plug to="/docs/installation/docker">
                  <Translate id="homepage.runtimeModes.plug.docker">Docker</Translate>
                </Plug>
              </ul>
              <footer className={styles.foot}>
                <Link className={duo.more} to="/docs/model-runtime/get-started/quickstart">
                  <Translate id="homepage.runtimeModes.engine.cta">Run the Engine</Translate>
                </Link>
                <Link className={duo.more} to="/docs/model-runtime/overview">
                  <Translate id="homepage.runtimeModes.engine.modelsCta">Explore the model runtime</Translate>
                </Link>
              </footer>
            </article>
          </div>

          <p className={clsx(duo.live, styles.note)}>
            <span className={duo.liveItem}>
              <i className={duo.liveDot} aria-hidden="true" />
              <Link className={styles.channel} to="/docs/installation">
                <Translate id="homepage.runtimeModes.devChannel">Development channel</Translate>
              </Link>
              <Translate id="homepage.runtimeModes.startupNote">Choose the composition at CLI startup.</Translate>
            </span>
          </p>
        </ScrollReveal>
      </div>
    </section>
  )
}
