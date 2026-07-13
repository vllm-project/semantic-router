import React from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
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
import { PillLink } from '@site/src/components/site/Chrome'
import TerrainCanvas from './TerrainCanvas'
import styles from './index.module.css'

const heroModelLogos = [
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

export default function SemanticTerrainHero(): JSX.Element {
  const logoSrc = useBaseUrl('/img/vllm-sr-logo.white.png')
  const modelCopies = [0, 1]
  const modelRepeats = [0, 1, 2]

  return (
    <section className={styles.stage}>
      <TerrainCanvas />

      <header className={styles.hero}>
        <div className={styles.heroScrim} aria-hidden="true" />
        <div className="site-shell-container">
          <div className={styles.copy}>
            <div className={styles.brand}>
              <img src={logoSrc} alt="vLLM Semantic Router" />
            </div>

            <h1>
              <span className={styles.accent}>
                <Translate id="homepage.hero.line1">Build your</Translate>
              </span>
              <span>
                <Translate id="homepage.hero.line2">
                  Mixture-of-Models
                </Translate>
              </span>
            </h1>

            <p className={styles.description}>
              <Translate id="homepage.hero.description">
                System-level intelligence for heterogeneous LLM inference
              </Translate>
            </p>

            <div className={styles.actions}>
              <PillLink
                className={styles.primaryCta}
                href="https://app.vllm-sr.ai/playground"
                rel="noreferrer"
                target="_blank"
              >
                <Translate id="homepage.hero.primaryCta">
                  Try the Playground
                </Translate>
              </PillLink>
              <PillLink
                className={styles.secondaryCta}
                to="/docs/intro"
                muted
              >
                <Translate id="homepage.hero.secondaryCta">
                  Explore the Docs
                </Translate>
              </PillLink>
            </div>
          </div>
        </div>

      </header>

      <section
        className={styles.modelBand}
        aria-label={translate({
          id: 'homepage.hero.modelBand.aria',
          message: 'Mixture-of-Models ecosystem',
        })}
      >
        <span className={styles.modelBandLabel}>
          <Translate id="homepage.hero.modelBand.eyebrow">
            Mixture-of-Models
          </Translate>
        </span>
        <div className={styles.modelViewport} aria-hidden="true">
          <div className={styles.modelTrack}>
            {modelCopies.map(copyIndex => (
              <div
                key={`terrain-models-${copyIndex}`}
                className={styles.modelSequence}
              >
                {modelRepeats.map(repeatIndex =>
                  heroModelLogos.map(({ label, Icon }) => (
                    <span
                      key={`${copyIndex}-${repeatIndex}-${label}`}
                      className={styles.model}
                    >
                      <Icon size={28} />
                      <strong>{label}</strong>
                    </span>
                  )),
                )}
              </div>
            ))}
          </div>
        </div>
      </section>
    </section>
  )
}
