import React, { useRef, useState } from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import { SectionLabel } from '@site/src/components/site/Chrome'
import styles from './styles.module.css'

type Video = {
  id: string
  title: string
  publisher: string
}

const FILM_DURATION = '2:55'

const videos: Video[] = [
  {
    id: 'Xow8Ns645sU',
    title: 'Enterprise AI Inference at Scale with AMD',
    publisher: 'AMD',
  },
  {
    id: 'QoHlqjSkNoo',
    title: '[vLLM Office Hours #52] - vLLM Semantic Router: Safer, Faster, Multi-Model Inference - June 25, 2026',
    publisher: 'Red Hat',
  },
  {
    id: 'b-ciRqvbtsk',
    title: '[vLLM Office Hours #34] AI-Powered vLLM Semantic Router - October 09, 2025',
    publisher: 'Red Hat',
  },
  {
    id: 'ExbMEW-Os1I',
    title: 'Inside open source AI strategy ft. Steve Watt | Technically Speaking with Chris Wright',
    publisher: 'Red Hat',
  },
  {
    id: '6SL27J7EyXM',
    title: 'Intelligent Query Routing using vLLM Semantic Router',
    publisher: 'NVIDIA Developer',
  },
  {
    id: 'A-gKzIJF6CA',
    title: 'vLLM Semantic Router: Intelligent Auto Reasoning for Efficient LLM Inference on Mixture-of-Models',
    publisher: 'Red Hat Open',
  },
  {
    id: 'xE0EJ6aYJng',
    title: 'Hybrid Inference in a box',
    publisher: 'Ricardo Noriega',
  },
]

function getEmbedUrl(videoId: string): string {
  return `https://www.youtube-nocookie.com/embed/${videoId}?rel=0&autoplay=1`
}

function getThumbnailUrl(videoId: string): string {
  return `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`
}

/* The project's own film. The poster is its first frame, so starting playback never jumps; preload="none" keeps the
 * film off the network until a visitor asks for it. */
function Film({ title }: { title: string }): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [started, setStarted] = useState(false)
  const src = useBaseUrl('/videos/vllm-sr-intro/vllm-sr-intro.mp4')
  const poster = useBaseUrl('/videos/vllm-sr-intro/vllm-sr-intro-poster.webp')

  return (
    <div className={styles.film}>
      <video
        ref={videoRef}
        className={styles.player}
        poster={poster}
        preload="none"
        playsInline
        controls={started}
        width={1920}
        height={1080}
        aria-label={title}
        onPlay={() => setStarted(true)}
        onEnded={() => {
          videoRef.current?.load()
          setStarted(false)
        }}
      >
        <source src={src} type="video/mp4" />
      </video>
      {!started && (
        <button
          type="button"
          className={styles.posterButton}
          onClick={() => {
            setStarted(true)
            videoRef.current?.play().catch(() => undefined)
          }}
        >
          <span className={styles.playPill}>
            <span className={styles.playIcon} aria-hidden="true">▶</span>
            <Translate id="homepage.hero.film.play">Watch the film</Translate>
            <span className={styles.duration}>{FILM_DURATION}</span>
          </span>
        </button>
      )}
    </div>
  )
}

/* After System Duo's console: the list on the left, the stage on the right. The film comes first. */
export default function YouTubeSection(): JSX.Element {
  const [active, setActive] = useState(-1)
  const [playingId, setPlayingId] = useState<string | null>(null)
  const filmPoster = useBaseUrl('/videos/vllm-sr-intro/vllm-sr-intro-poster.webp')
  const filmTitle = translate({ id: 'homepage.hero.film.title', message: 'Reintroducing vLLM Semantic Router' })
  const activeVideo = active >= 0 ? videos[active] : null

  const select = (index: number) => {
    setPlayingId(null)
    setActive(index)
  }

  return (
    <section id="watch" className={styles.section} aria-labelledby="video-showcase-title">
      <div className={styles.wrap}>
        <header className="site-section-intro">
          <SectionLabel>
            <Translate id="homepage.videos.label">See it in action</Translate>
          </SectionLabel>
          <h2 id="video-showcase-title">
            <Translate id="homepage.videos.title">Watch it work.</Translate>
          </h2>
          <p>
            <Translate id="homepage.videos.description">
              Demos, deployments, and ideas from the community.
            </Translate>
          </p>
        </header>

        <div className={styles.console}>
          <ul
            className={styles.list}
            aria-label={translate({ id: 'homepage.videos.rail.aria', message: 'Video playlist' })}
          >
            <li>
              <button
                type="button"
                className={styles.card}
                aria-pressed={active === -1}
                onClick={() => select(-1)}
              >
                <img src={filmPoster} alt="" loading="lazy" width="480" height="270" />
                <span className={styles.cardCopy}>
                  <strong>{filmTitle}</strong>
                  <span>
                    vLLM Semantic Router
                    {' · '}
                    {FILM_DURATION}
                  </span>
                </span>
              </button>
            </li>
            {videos.map((video, index) => (
              <li key={video.id}>
                <button
                  type="button"
                  className={styles.card}
                  aria-pressed={active === index}
                  aria-label={translate(
                    { id: 'homepage.videos.play.aria', message: 'Show video: {title}' },
                    { title: video.title },
                  )}
                  onClick={() => select(index)}
                >
                  <img src={getThumbnailUrl(video.id)} alt="" loading="lazy" width="480" height="360" />
                  <span className={styles.cardCopy}>
                    <strong>{video.title}</strong>
                    <span>{video.publisher}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>

          <article className={styles.stage}>
            <div className={styles.frame}>
              {!activeVideo && <Film title={filmTitle} />}
              {activeVideo && playingId === activeVideo.id && (
                <iframe
                  key={activeVideo.id}
                  className={styles.player}
                  src={getEmbedUrl(activeVideo.id)}
                  title={activeVideo.title}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                  referrerPolicy="strict-origin-when-cross-origin"
                  allowFullScreen
                />
              )}
              {activeVideo && playingId !== activeVideo.id && (
                <button
                  type="button"
                  className={styles.posterButton}
                  aria-label={translate(
                    { id: 'homepage.videos.playFeatured.aria', message: 'Play video: {title}' },
                    { title: activeVideo.title },
                  )}
                  onClick={() => setPlayingId(activeVideo.id)}
                >
                  <img className={styles.player} src={getThumbnailUrl(activeVideo.id)} alt="" width="480" height="360" />
                  <span className={styles.playPill}>
                    <span className={styles.playIcon} aria-hidden="true">▶</span>
                    <Translate id="homepage.videos.playFeatured">Play video</Translate>
                  </span>
                </button>
              )}
            </div>
            <div className={styles.caption}>
              <span className={styles.kind}>
                {active === -1
                  ? <Translate id="homepage.videos.featured">Featured</Translate>
                  : <Translate id="homepage.videos.selected">Selected video</Translate>}
              </span>
              <h3>{activeVideo ? activeVideo.title : filmTitle}</h3>
              <span className={styles.publisher}>{activeVideo ? activeVideo.publisher : 'vLLM Semantic Router'}</span>
            </div>
          </article>
        </div>
      </div>
    </section>
  )
}
