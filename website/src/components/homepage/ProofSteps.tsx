import React from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { PillLink } from '@site/src/components/site/Chrome'
import shared from './homepageShared.module.css'
import styles from './ProofSteps.module.css'

function buildInstallScriptUrl(siteUrl: string, baseUrl: string): string {
  const normalizedSiteUrl = siteUrl.replace(/\/$/, '')
  const normalizedBaseUrl = baseUrl === '/' ? '' : baseUrl.replace(/\/$/, '')
  return `${normalizedSiteUrl}${normalizedBaseUrl}/install.sh`
}

const challengeItems = [
  translate({
    id: 'homepage.proof.challenge1',
    message: '16 signal families — one config surface',
  }),
  translate({
    id: 'homepage.proof.challenge2',
    message: 'Router replay — full decision audit',
  }),
  translate({
    id: 'homepage.proof.challenge3',
    message: 'Mixture-of-Models — heterogeneous fleet routing',
  }),
  translate({
    id: 'homepage.proof.challenge4',
    message: 'Envoy ext_proc — drop-in inference gateway',
  }),
]

export default function ProofSteps(): JSX.Element {
  const { siteConfig } = useDocusaurusContext()
  const installScriptUrl = buildInstallScriptUrl(siteConfig.url, siteConfig.baseUrl)
  const installCommand = `curl -fsSL ${installScriptUrl} | bash`

  const steps = [
    {
      number: '01',
      title: translate({
        id: 'homepage.proof.step1.title',
        message: 'Download & install',
      }),
      description: translate({
        id: 'homepage.proof.step1.desc',
        message: 'One script, local dev image, and vllm-sr CLI. Running in minutes.',
      }),
      code: installCommand,
    },
    {
      number: '02',
      title: translate({
        id: 'homepage.proof.step2.title',
        message: 'Configure routing',
      }),
      description: translate({
        id: 'homepage.proof.step2.desc',
        message: 'Define decisions, signals, algorithms, and plugins in one YAML contract.',
      }),
      code: 'vllm-sr serve --image-pull-policy never',
    },
    {
      number: '03',
      title: translate({
        id: 'homepage.proof.step3.title',
        message: 'Route a request',
      }),
      description: translate({
        id: 'homepage.proof.step3.desc',
        message: 'Send OpenAI-compatible traffic and inspect x-vsr-replay-id on every response.',
      }),
      code: 'curl localhost:8080/v1/chat/completions ...',
    },
    {
      number: '04',
      title: translate({
        id: 'homepage.proof.step4.title',
        message: 'Explore docs',
      }),
      description: translate({
        id: 'homepage.proof.step4.desc',
        message: 'Tutorials, signal reference, and operator guides to go deeper.',
      }),
      code: 'vllm-sr.ai/docs',
    },
  ]

  return (
    <section className={shared.lightSection} aria-labelledby="proof-steps-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <header className={shared.sectionHeader}>
          <span className={shared.sectionEyebrow}>
            <Translate id="homepage.proof.eyebrow">Seeing is believing</Translate>
          </span>
          <h2 id="proof-steps-title" className={shared.sectionTitle}>
            <Translate id="homepage.proof.title">
              Do not take our word for it. Prove it yourself.
            </Translate>
          </h2>
          <p className={shared.sectionSubtitle}>
            <Translate id="homepage.proof.subtitle">
              In fifteen minutes you can install, route a request, and inspect the
              replay record behind every decision.
            </Translate>
          </p>
        </header>

        <div className={styles.stepGrid}>
          {steps.map(step => (
            <article key={step.number} className={styles.stepCard}>
              <span className={styles.stepNumber}>{step.number}</span>
              <h3>{step.title}</h3>
              <p>{step.description}</p>
              <pre className={styles.codeBlock}>
                <code>{step.code}</code>
              </pre>
            </article>
          ))}
        </div>

        <div className={styles.challengePanel}>
          <h3>
            <Translate id="homepage.proof.challengeTitle">Take the challenge</Translate>
          </h3>
          <ul className={styles.challengeList}>
            {challengeItems.map(item => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>

        <div className={styles.ctaRow}>
          <PillLink to="/docs/installation">
            <Translate id="homepage.proof.primaryCta">Get started now</Translate>
          </PillLink>
          <PillLink href="https://github.com/vllm-project/semantic-router" rel="noreferrer" target="_blank" muted>
            <Translate id="homepage.proof.secondaryCta">Star on GitHub</Translate>
          </PillLink>
        </div>
      </div>
    </section>
  )
}
