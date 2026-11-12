import React from 'react'
import clsx from 'clsx'
import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import Claude from '@lobehub/icons/es/Claude/components/Mono'
import DeepSeek from '@lobehub/icons/es/DeepSeek/components/Mono'
import Gemini from '@lobehub/icons/es/Gemini/components/Mono'
import Meta from '@lobehub/icons/es/Meta/components/Mono'
import OpenAI from '@lobehub/icons/es/OpenAI/components/Mono'
import Qwen from '@lobehub/icons/es/Qwen/components/Mono'
import { FiCheck, FiDatabase, FiGlobe, FiKey, FiLock, FiMapPin, FiShield } from 'react-icons/fi'
import { SectionLabel } from '@site/src/components/site/Chrome'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import duo from '@site/src/components/duo/duo.module.css'
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

function Marks({ group }: { group: keyof typeof modelGroups }): React.JSX.Element {
  return (
    <ul className={duo.marks}>
      {modelGroups[group].map(({ name, Icon }) => (
        <li key={name}>
          <Icon size={18} aria-hidden="true" />
          {name}
        </li>
      ))}
    </ul>
  )
}

function Tag({ n, children }: { n: string, children: React.ReactNode }): React.JSX.Element {
  return (
    <span className={styles.tag}>
      <b>{n}</b>
      {children}
    </span>
  )
}

/* Privacy-aware routing as System Duo's full stack: plates that widen towards the GPUs, from the request down. */
export default function SovereigntyAI(): React.JSX.Element {
  const mark = useBaseUrl('/img/vllm-sr-mark.svg')

  return (
    <section id="sovereignty-ai" className={styles.section} aria-labelledby="sovereignty-ai-title">
      <div className={styles.wrap}>
        <ScrollReveal>
          <header className="site-section-intro">
            <SectionLabel>
              <Translate id="homepage.sovereigntyAI.label">Sovereignty AI</Translate>
            </SectionLabel>
            <h2 id="sovereignty-ai-title">
              <Translate id="homepage.sovereigntyAI.title">Your knowledge. Your control.</Translate>
            </h2>
            <p>
              <Translate id="homepage.sovereigntyAI.description">
                Private knowledge stays on your GPUs. Frontier models only when your policy allows.
              </Translate>
            </p>
          </header>
        </ScrollReveal>

        <ScrollReveal delay={80}>
          <figure className={styles.figure}>
            <ol className={styles.slabs}>
              <li className={styles.slab}>
                <Tag n="01"><Translate id="homepage.sovereigntyAI.incomingRequest">Incoming request</Translate></Tag>
                <div className={styles.main}>
                  <p className={styles.request}>
                    <FiDatabase aria-hidden="true" />
                    <Translate id="homepage.sovereigntyAI.privateRequest">Internal domain knowledge</Translate>
                    <span className={styles.verdict}>
                      <FiLock aria-hidden="true" />
                      <Translate id="homepage.sovereigntyAI.keepOnPrem">Keep on-prem</Translate>
                    </span>
                  </p>
                  <p className={styles.request}>
                    <FiGlobe aria-hidden="true" />
                    <Translate id="homepage.sovereigntyAI.publicRequest">Public research</Translate>
                    <span className={clsx(styles.verdict, styles.verdictSky)}>
                      <FiCheck aria-hidden="true" />
                      <Translate id="homepage.sovereigntyAI.externalEligible">External eligible</Translate>
                    </span>
                  </p>
                </div>
              </li>

              <li className={clsx(styles.slab, styles.gate)}>
                <Tag n="02"><Translate id="homepage.sovereigntyAI.gate">Policy gate</Translate></Tag>
                <div className={styles.main}>
                  <h3>
                    <img src={mark} alt="" width={26} height={26} />
                    <Translate id="homepage.sovereigntyAI.visualTitle">Privacy-aware routing</Translate>
                  </h3>
                  <ul className={duo.chips}>
                    <li>
                      <FiShield aria-hidden="true" />
                      <Translate id="homepage.sovereigntyAI.gate.privacy">Privacy</Translate>
                    </li>
                    <li>
                      <FiMapPin aria-hidden="true" />
                      <Translate id="homepage.sovereigntyAI.gate.locality">Locality</Translate>
                    </li>
                    <li>
                      <FiKey aria-hidden="true" />
                      <Translate id="homepage.sovereigntyAI.gate.access">Access</Translate>
                    </li>
                  </ul>
                  <span className={styles.policy}>
                    <i aria-hidden="true" />
                    <Translate id="homepage.sovereigntyAI.policyControl">Policy in control</Translate>
                  </span>
                </div>
              </li>

              <li className={styles.slab}>
                <Tag n="03"><Translate id="homepage.sovereigntyAI.eligibilityFirst">Eligibility first</Translate></Tag>
                <div className={styles.main}>
                  <h3><Translate id="homepage.sovereigntyAI.selectionSecond">Model selection</Translate></h3>
                  <p><Translate id="homepage.sovereigntyAI.objectives">Quality · latency · cost</Translate></p>
                </div>
              </li>

              <li className={styles.slab}>
                <Tag n="04"><Translate id="homepage.sovereigntyAI.models">Models</Translate></Tag>
                <div className={styles.halves}>
                  <div className={styles.half}>
                    <b><Translate id="homepage.sovereigntyAI.openModels">Open models</Translate></b>
                    <span><Translate id="homepage.sovereigntyAI.onPremGPU">On-prem GPU</Translate></span>
                    <Marks group="open" />
                  </div>
                  <div className={clsx(styles.half, styles.halfSky)}>
                    <b><Translate id="homepage.sovereigntyAI.closedModels">Frontier closed models</Translate></b>
                    <span>
                      <Translate id="homepage.sovereigntyAI.externalProvider">External provider</Translate>
                      {' · '}
                      <Translate id="homepage.sovereigntyAI.externalCondition">Only policy-eligible requests</Translate>
                    </span>
                    <Marks group="closed" />
                  </div>
                </div>
              </li>

              <li className={clsx(styles.slab, styles.hardware)}>
                <Tag n="05"><Translate id="homepage.sovereigntyAI.hardware">Hardware</Translate></Tag>
                <div className={styles.main}>
                  <div>
                    <h3><Translate id="homepage.sovereigntyAI.localCompute">On-prem GPUs. Open models.</Translate></h3>
                    <p><Translate id="homepage.sovereigntyAI.knowledgeOnPrem">Domain knowledge stays on-prem</Translate></p>
                  </div>
                </div>
              </li>
            </ol>
            <figcaption className={styles.srOnly}>
              <Translate id="homepage.sovereigntyAI.diagramDescription">
                This diagram illustrates a configured privacy policy: internal domain knowledge is routed to open models on on-prem GPUs, while public requests can use external frontier closed models when eligible. Privacy, locality, and access constraints apply before model selection.
              </Translate>
            </figcaption>
          </figure>

          <p className={styles.cta}>
            <Link className={duo.more} to="/docs/overview/signal-driven-decisions">
              <Translate id="homepage.sovereigntyAI.cta">Explore routing policies</Translate>
            </Link>
          </p>
        </ScrollReveal>
      </div>
    </section>
  )
}
