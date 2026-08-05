import React from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import { PillLink, SectionLabel } from '@site/src/components/site/Chrome'
import shared from './homepageShared.module.css'
import styles from './EncoderTeaser.module.css'

const tracks = [
  {
    label: 'SEQ_CLS',
    text: translate({
      id: 'homepage.encoderTeaser.track.sequence',
      message: 'Domain, jailbreak, and feedback routing',
    }),
  },
  {
    label: 'TOKEN',
    text: translate({
      id: 'homepage.encoderTeaser.track.token',
      message: 'PII and safety-sensitive spans',
    }),
  },
  {
    label: 'EMBED',
    text: translate({
      id: 'homepage.encoderTeaser.track.embedding',
      message: 'Semantic cache, KB routing, reranking',
    }),
  },
]

export default function EncoderTeaser(): JSX.Element {
  return (
    <section className={shared.lightSection} aria-labelledby="encoder-teaser-title">
      <div className={`site-shell-container ${shared.sectionInner}`}>
        <div className={styles.layout}>
          <header className={styles.copy}>
            <SectionLabel>
              <Translate id="homepage.encoderTeaser.label">Signal intelligence</Translate>
            </SectionLabel>
            <h2 id="encoder-teaser-title" className={shared.sectionTitle}>
              <Translate id="homepage.encoderTeaser.title">Encoder signals for routing</Translate>
            </h2>
            <p className={shared.sectionSubtitle}>
              <Translate id="homepage.encoderTeaser.description">Classify intent, label sensitive spans, and score relevance before generation starts.</Translate>
            </p>
            <PillLink
              href="https://huggingface.co/LLM-Semantic-Router"
              rel="noreferrer"
              target="_blank"
            >
              <Translate id="homepage.encoderTeaser.cta">Hugging Face models</Translate>
            </PillLink>
          </header>

          <ul className={styles.trackList}>
            {tracks.map(track => (
              <li key={track.label} className={styles.track}>
                <span className={styles.trackLabel}>{track.label}</span>
                <span>{track.text}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  )
}
