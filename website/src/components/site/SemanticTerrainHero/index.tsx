import React, { useRef, useState } from 'react'
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

const FILM_DURATION = '2:55'

/* The poster is the film's first frame, so starting playback never jumps.
 * preload="none" keeps the film off the network until a visitor asks for it. */
function HeroFilm(): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [started, setStarted] = useState(false)
  const src = useBaseUrl('/videos/vllm-sr-intro/vllm-sr-intro.mp4')
  const poster = useBaseUrl('/videos/vllm-sr-intro/vllm-sr-intro-poster.webp')
  const title = translate({
    id: 'homepage.hero.film.title',
    message: 'Reintroducing vLLM Semantic Router',
  })

  const start = () => {
    setStarted(true)
    videoRef.current?.play().catch(() => undefined)
  }

  const reset = () => {
    videoRef.current?.load()
    setStarted(false)
  }

  return (
    <figure className={styles.film}>
      <video
        ref={videoRef}
        className={styles.filmVideo}
        poster={poster}
        preload="none"
        playsInline
        controls={started}
        width={1920}
        height={1080}
        aria-label={title}
        onPlay={() => setStarted(true)}
        onEnded={reset}
      >
        <source src={src} type="video/mp4" />
      </video>
      {!started && (
        <button type="button" className={styles.filmPlay} onClick={start}>
          <span className={styles.filmPlayPill}>
            <span className={styles.filmPlayIcon} aria-hidden="true">
              <svg viewBox="0 0 24 24" focusable="false">
                <path d="M8 5.6v12.8a.6.6 0 0 0 .92.5l10.1-6.4a.6.6 0 0 0 0-1L8.92 5.1A.6.6 0 0 0 8 5.6Z" />
              </svg>
            </span>
            <Translate id="homepage.hero.film.play">Watch the film</Translate>
            <span className={styles.filmDuration}>{FILM_DURATION}</span>
          </span>
        </button>
      )}
    </figure>
  )
}

export default function SemanticTerrainHero(): JSX.Element {
  const modelCopies = [0, 1]
  const modelRepeats = [0, 1, 2]

  return (
    <section className={styles.stage}>
      <div className={styles.heroBackdrop} aria-hidden="true">
        <span className={styles.heroGlow} data-glow="a" />
        <span className={styles.heroGlow} data-glow="b" />
        <span className={styles.heroGlow} data-glow="c" />
        <span className={styles.heroGrid} />
      </div>

      <header className={styles.hero}>
        <div className="site-shell-container">
          <div className={styles.heroInner}>
            <div className={styles.intro}>
              <div className={styles.introCopy}>
                <h1 className={styles.title}>
                  <Translate id="homepage.hero.line1">Make your</Translate>
                  {' '}
                  <span className={`${styles.accent} ${styles.nowrap}`}>
                    <Translate id="homepage.hero.line2">Mixture-of-Models</Translate>
                  </span>
                  {' '}
                  <Translate id="homepage.hero.line3">programmable.</Translate>
                </h1>
                <p className={styles.dek}>
                  <Translate id="homepage.hero.dek">
                    The right model, on the right compute, for every request.
                  </Translate>
                </p>
              </div>
              <div className={styles.actions}>
                <PillLink
                  className={styles.primaryCta}
                  href="https://app.vllm-sr.ai/playground"
                >
                  <Translate id="homepage.hero.primaryCta">
                    Try Playground
                  </Translate>
                  <span aria-hidden="true">→</span>
                </PillLink>
                <PillLink className={styles.secondaryCta} to="/docs/intro" muted>
                  <Translate id="homepage.hero.secondaryCta">
                    Read the Docs
                  </Translate>
                  <span aria-hidden="true">→</span>
                </PillLink>
              </div>
            </div>
            <HeroFilm />
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
