import React, { useRef } from 'react'
import clsx from 'clsx'
import Translate, { translate } from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import ThemedImage from '@theme/ThemedImage'
import Claude from '@lobehub/icons/es/Claude/components/Mono'
import DeepSeek from '@lobehub/icons/es/DeepSeek/components/Mono'
import Gemini from '@lobehub/icons/es/Gemini/components/Mono'
import Grok from '@lobehub/icons/es/Grok/components/Mono'
import Kimi from '@lobehub/icons/es/Kimi/components/Mono'
import Meta from '@lobehub/icons/es/Meta/components/Mono'
import Minimax from '@lobehub/icons/es/Minimax/components/Mono'
import Mistral from '@lobehub/icons/es/Mistral/components/Mono'
import OpenAI from '@lobehub/icons/es/OpenAI/components/Mono'
import Qwen from '@lobehub/icons/es/Qwen/components/Mono'
import Zhipu from '@lobehub/icons/es/Zhipu/components/Mono'
import { SectionLabel } from '@site/src/components/site/Chrome'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import PixelIcon, { type PixelIconName } from '@site/src/components/duo/PixelIcon'
import usePauseOffscreen from '@site/src/components/duo/usePauseOffscreen'
import duo from '@site/src/components/duo/duo.module.css'
import styles from './Architecture.module.css'

const SIGNALS = ['domain', 'complexity', 'jailbreak', 'pii', 'language', 'context', 'keyword']

const MODEL_MARKS = [
  { label: 'Kimi', Icon: Kimi },
  { label: 'Zhipu', Icon: Zhipu },
  { label: 'MiniMax', Icon: Minimax },
  { label: 'ChatGPT', Icon: OpenAI },
  { label: 'Claude', Icon: Claude },
  { label: 'Gemini', Icon: Gemini },
  { label: 'DeepSeek', Icon: DeepSeek },
  { label: 'Qwen', Icon: Qwen },
  { label: 'Llama', Icon: Meta },
  { label: 'Mistral', Icon: Mistral },
  { label: 'Grok', Icon: Grok },
]

function Node({ icon, tone, label, title, children }: {
  icon: PixelIconName
  tone?: 'one' | 'two'
  label: React.ReactNode
  title: React.ReactNode
  children?: React.ReactNode
}): JSX.Element {
  return (
    <div className={clsx(styles.node, tone === 'one' && styles.one, tone === 'two' && styles.two)}>
      <PixelIcon name={icon} size={24} className={styles.icon} />
      <span className={styles.label}>{label}</span>
      <h3>{title}</h3>
      {children}
    </div>
  )
}

function Wire({ delay = 0, slow = false }: { delay?: 0 | 1 | 2 | 3, slow?: boolean }): JSX.Element {
  return (
    <span className={clsx(styles.wire, slow && styles.slow, delay && styles[`d${delay}`])} aria-hidden="true">
      <i />
    </span>
  )
}

function RouterBox({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <div className={styles.router}>
      <span className={styles.tag}>
        <ThemedImage
          sources={{ light: useBaseUrl('/img/vllm-sr-logo.light.png'), dark: useBaseUrl('/img/vllm-sr-logo.white.png') }}
          alt="vLLM Semantic Router"
          height={28}
        />
      </span>
      {children}
    </div>
  )
}

/* After System Duo's Architecture: the two ways a request goes through vLLM Semantic Router, read left to right. Router
 * mode routes every chat request; Engine mode answers a direct question. Squares relay along each row. */
export default function Architecture(): JSX.Element {
  const ref = useRef<HTMLElement>(null)
  usePauseOffscreen(ref)

  return (
    <section ref={ref} id="architecture" className={styles.section} aria-labelledby="architecture-title">
      <div className={styles.wrap}>
        <ScrollReveal>
          <header className="site-section-intro">
            <SectionLabel>
              <Translate id="homepage.capabilities.label">Architecture</Translate>
            </SectionLabel>
            <h2 id="architecture-title">
              <Translate id="homepage.capabilities.heading">Your models. Your rules.</Translate>
            </h2>
            <p>
              <Translate id="homepage.arch.lede">
                One frontend, two modes: the Router routes chat, the Engine answers questions directly.
              </Translate>
            </p>
          </header>
        </ScrollReveal>

        <ScrollReveal delay={80}>
          <figure className={styles.figure}>
            <div className={styles.row} role="group" aria-labelledby="arch-router-row">
              <p className={styles.rowLabel} id="arch-router-row">
                <i className={clsx(duo.sysDot, duo.sysDotSky)} aria-hidden="true" />
                <Translate id="homepage.arch.routerRow">Router mode · every chat request</Translate>
              </p>
              <Node
                icon="client"
                label={<Translate id="homepage.arch.request.label">Your app or agent</Translate>}
                title={<Translate id="homepage.arch.request.title">A request</Translate>}
              >
                <p>
                  <Translate
                    id="homepage.arch.request.body"
                    values={{ chat: <code>/v1/chat/completions</code>, messages: <code>/v1/messages</code> }}
                  >
                    {'Any OpenAI or Anthropic SDK: {chat}, {messages}'}
                  </Translate>
                </p>
              </Node>
              <Wire />
              <RouterBox>
                <Node
                  icon="model"
                  tone="one"
                  label={<Translate id="homepage.arch.signals.label">Signals</Translate>}
                  title={<Translate id="homepage.arch.signals.title">Read the request</Translate>}
                >
                  <ul
                    className={styles.signals}
                    aria-label={translate({ id: 'homepage.arch.signals.aria', message: 'Some of the signals' })}
                  >
                    {SIGNALS.map(signal => <li key={signal}>{signal}</li>)}
                  </ul>
                </Node>
                <Wire delay={1} />
                <Node
                  icon="engine"
                  label={<Translate id="homepage.arch.decision.label">Decision</Translate>}
                  title={<Translate id="homepage.arch.decision.title">Pick the route</Translate>}
                >
                  <p>
                    <Translate id="homepage.arch.decision.body">
                      Projections and decisions apply your policy; an algorithm picks the model.
                    </Translate>
                  </p>
                </Node>
                <Wire delay={2} />
                <Node
                  icon="front"
                  label={<Translate id="homepage.arch.plugins.label">Plugins</Translate>}
                  title={<Translate id="homepage.arch.plugins.title">Send it on</Translate>}
                >
                  <p>
                    <Translate id="homepage.arch.plugins.body">
                      Prompts, memory, caching and safety; streaming and routing headers.
                    </Translate>
                  </p>
                </Node>
              </RouterBox>
              <Wire slow />
              <Node
                icon="pool"
                tone="two"
                label={<Translate id="homepage.arch.models.label">Your models</Translate>}
                title={<Translate id="homepage.arch.models.title">Run it anywhere</Translate>}
              >
                <p>
                  <Translate id="homepage.arch.models.body">
                    vLLM, SGLang or any OpenAI-compatible API: cloud, private or edge.
                  </Translate>
                </p>
                <ul className={styles.marks} aria-label={translate({ id: 'homepage.arch.models.aria', message: 'Models it routes to' })}>
                  {MODEL_MARKS.map(({ label, Icon }) => (
                    <li key={label} title={label}>
                      <Icon size={16} aria-hidden="true" />
                      <span className={styles.srOnly}>{label}</span>
                    </li>
                  ))}
                </ul>
              </Node>
            </div>

            <div className={styles.row} role="group" aria-labelledby="arch-engine-row">
              <p className={styles.rowLabel} id="arch-engine-row">
                <i className={duo.sysDot} aria-hidden="true" />
                <Translate id="homepage.arch.engineRow">Engine mode · a direct question</Translate>
              </p>
              <Node
                icon="client"
                label={<Translate id="homepage.arch.agent.label">Your agent</Translate>}
                title={<Translate id="homepage.arch.agent.title">A small question</Translate>}
              >
                <p>
                  <Translate id="homepage.arch.agent.body" values={{ endpoint: <code>POST /v1/systemone</code> }}>
                    {'{endpoint}: risky? Which team? Done?'}
                  </Translate>
                </p>
              </Node>
              <Wire delay={1} />
              <RouterBox>
                <Node
                  icon="model"
                  tone="one"
                  label={<Translate id="homepage.arch.systemOne.label">System One</Translate>}
                  title={<Translate id="homepage.arch.systemOne.title">Decision models</Translate>}
                >
                  <p>
                    <Translate id="homepage.arch.systemOne.body">
                      Vela 3.0 and Decision 3.0, on CPU or GPU: text, images and video.
                    </Translate>
                  </p>
                </Node>
                <Wire delay={2} />
                <Node
                  icon="bolt"
                  label={<Translate id="homepage.arch.pass.label">One forward pass</Translate>}
                  title={<Translate id="homepage.arch.pass.title">Every answer</Translate>}
                >
                  <p>
                    <Translate id="homepage.arch.pass.body">A probability for each option, no generated text.</Translate>
                  </p>
                </Node>
                <Wire delay={3} />
                <Node
                  icon="dial"
                  label={<Translate id="homepage.arch.calibrated.label">Calibrated</Translate>}
                  title={<Translate id="homepage.arch.calibrated.title">Act on it</Translate>}
                >
                  <p>
                    <Translate id="homepage.arch.calibrated.body">Allow, ask, route or stop, at your threshold.</Translate>
                  </p>
                </Node>
              </RouterBox>
              <Wire />
              <Node
                icon="open"
                label={<Translate id="homepage.arch.back.label">Back to you</Translate>}
                title={<Translate id="homepage.arch.back.title">Your next step</Translate>}
              >
                <p>
                  <Translate id="homepage.arch.back.body">
                    Your application acts on the answer; its LLM keeps its tokens for the work.
                  </Translate>
                </p>
              </Node>
            </div>

            <figcaption className={styles.srOnly}>
              <Translate id="homepage.arch.caption">
                In Router mode, a chat request is read for signals, matched to a decision whose algorithm picks the model, passed through plugins and sent to your models. In Engine mode, a direct question to /v1/systemone is answered by a decision model in one forward pass, with a calibrated probability for each option.
              </Translate>
            </figcaption>
          </figure>

        </ScrollReveal>
      </div>
    </section>
  )
}
