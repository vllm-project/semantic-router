import React from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import ScrollReveal from '@site/src/components/site/ScrollReveal'
import shared from './homepageShared.module.css'
import styles from './EcosystemStrip.module.css'

const ecosystemLinks = [
  {
    name: 'vLLM',
    description: translate({
      id: 'homepage.ecosystem.vllm',
      message: 'High-throughput LLM serving',
    }),
    href: 'https://vllm.ai/',
  },
  {
    name: 'AIBrix',
    description: translate({
      id: 'homepage.ecosystem.aibrix',
      message: 'AI infrastructure on Kubernetes',
    }),
    href: 'https://github.com/vllm-project/aibrix',
  },
  {
    name: 'Production Stack',
    description: translate({
      id: 'homepage.ecosystem.production',
      message: 'Production deployment recipes',
    }),
    href: 'https://github.com/vllm-project/production-stack',
  },
  {
    name: 'GuideLLM',
    description: translate({
      id: 'homepage.ecosystem.guidellm',
      message: 'LLM performance evaluation',
    }),
    href: 'https://github.com/vllm-project/guidellm',
  },
  {
    name: 'Speculators',
    description: translate({
      id: 'homepage.ecosystem.speculators',
      message: 'Speculative decoding toolkit',
    }),
    href: 'https://github.com/vllm-project/speculators',
  },
  {
    name: 'Playground',
    description: translate({
      id: 'homepage.ecosystem.playground',
      message: 'Try routing live',
    }),
    href: 'https://app.vllm-sr.ai/playground',
    highlight: true,
  },
]

export default function EcosystemStrip(): JSX.Element {
  return (
    <section className={styles.section} aria-labelledby="ecosystem-title">
      <div className="site-shell-container">
        <ScrollReveal>
          <header className={styles.header}>
            <h2 id="ecosystem-title" className={shared.sectionTitle}>
              <Translate id="homepage.ecosystem.title">vLLM ecosystem</Translate>
            </h2>
            <p className={shared.sectionSubtitle}>
              <Translate id="homepage.ecosystem.subtitle">Tools and libraries built around the vLLM stack.</Translate>
            </p>
          </header>
        </ScrollReveal>

        <div className={styles.grid}>
          {ecosystemLinks.map((link, index) => (
            <ScrollReveal key={link.name} delay={index * 50}>
              <a
                href={link.href}
                className={link.highlight ? styles.cardHighlight : styles.card}
                rel="noreferrer"
                target="_blank"
              >
                <strong>{link.name}</strong>
                <span>{link.description}</span>
              </a>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  )
}
