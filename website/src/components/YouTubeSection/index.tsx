import React, { useState } from 'react'
import Translate, { translate } from '@docusaurus/Translate'
import { SectionLabel } from '@site/src/components/site/Chrome'
import styles from './styles.module.css'

type Video = {
  id: string
  title: string
  publisher: string
}

const VIDEOS_PER_PAGE = 6

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
  return `https://www.youtube-nocookie.com/embed/${videoId}?rel=0`
}

function getThumbnailUrl(videoId: string): string {
  return `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`
}

export default function YouTubeSection(): JSX.Element {
  const [activeIndex, setActiveIndex] = useState(0)
  const activeVideo = videos[activeIndex]
  const pageCount = Math.ceil(videos.length / VIDEOS_PER_PAGE)
  const activePage = Math.floor(activeIndex / VIDEOS_PER_PAGE)
  const pageStart = activePage * VIDEOS_PER_PAGE
  const visibleVideos = videos.slice(pageStart, pageStart + VIDEOS_PER_PAGE)

  const selectRelativeVideo = (offset: number) => {
    setActiveIndex(current => (current + offset + videos.length) % videos.length)
  }

  const selectRelativePage = (offset: number) => {
    const nextPage = (activePage + offset + pageCount) % pageCount
    setActiveIndex(nextPage * VIDEOS_PER_PAGE)
  }

  return (
    <section className={styles.section} aria-labelledby="video-showcase-title">
      <div className="site-shell-container">
        <header className={`site-section-intro ${styles.heading}`}>
          <SectionLabel>
            <Translate id="homepage.videos.label">See it in action</Translate>
          </SectionLabel>
          <h2 id="video-showcase-title">
            <Translate id="homepage.videos.title">Semantic routing in the real world</Translate>
          </h2>
          <p>
            <Translate id="homepage.videos.description">
              See how teams use semantic routing across enterprise inference,
              open model serving, hybrid systems, and Mixture-of-Models.
            </Translate>
          </p>
        </header>

        <div className={styles.showcase}>
          <article className={styles.featuredVideo}>
            <div className={styles.videoMeta}>
              <span className={styles.videoKind}>
                {activeIndex === 0
                  ? <Translate id="homepage.videos.featured">Featured</Translate>
                  : <Translate id="homepage.videos.selected">Selected video</Translate>}
              </span>
              <span className={styles.publisher}>{activeVideo.publisher}</span>
              <h3>{activeVideo.title}</h3>
            </div>

            <div className={styles.playerFrame}>
              <iframe
                key={activeVideo.id}
                className={styles.player}
                src={getEmbedUrl(activeVideo.id)}
                title={activeVideo.title}
                loading="lazy"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                referrerPolicy="strict-origin-when-cross-origin"
                allowFullScreen
              />
            </div>

            <div className={styles.playerControls}>
              <button
                type="button"
                className={styles.directionButton}
                onClick={() => {
                  selectRelativeVideo(-1)
                }}
                aria-label={translate({
                  id: 'homepage.videos.previous.aria',
                  message: 'Show previous video',
                })}
              >
                <span aria-hidden="true">←</span>
                <Translate id="homepage.videos.previous">Previous</Translate>
              </button>
              <span className={styles.videoCount} aria-live="polite">
                {activeIndex + 1}
                {' / '}
                {videos.length}
              </span>
              <button
                type="button"
                className={styles.directionButton}
                onClick={() => {
                  selectRelativeVideo(1)
                }}
                aria-label={translate({
                  id: 'homepage.videos.next.aria',
                  message: 'Show next video',
                })}
              >
                <Translate id="homepage.videos.next">Next</Translate>
                <span aria-hidden="true">→</span>
              </button>
            </div>
          </article>

          <ul
            className={styles.videoRail}
            aria-label={translate({
              id: 'homepage.videos.rail.aria',
              message: 'Video playlist',
            })}
          >
            {visibleVideos.map((video, index) => {
              const videoIndex = pageStart + index
              const active = videoIndex === activeIndex

              return (
                <li key={video.id} className={styles.videoRailItem}>
                  <button
                    type="button"
                    className={`${styles.videoCard} ${active ? styles.videoCardActive : ''}`}
                    aria-pressed={active}
                    aria-label={translate(
                      {
                        id: 'homepage.videos.play.aria',
                        message: 'Show video: {title}',
                      },
                      { title: video.title },
                    )}
                    onClick={() => {
                      setActiveIndex(videoIndex)
                    }}
                  >
                    <span className={styles.thumbnail}>
                      <img
                        src={getThumbnailUrl(video.id)}
                        alt=""
                        loading="lazy"
                        width="480"
                        height="360"
                      />
                      <span className={styles.playMark} aria-hidden="true">▶</span>
                    </span>
                    <span className={styles.cardCopy}>
                      <strong>{video.title}</strong>
                      <span>{video.publisher}</span>
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>

          {pageCount > 1 && (
            <nav
              className={styles.videoPagination}
              aria-label={translate({
                id: 'homepage.videos.pagination.aria',
                message: 'Video playlist pages',
              })}
            >
              <button
                type="button"
                className={styles.directionButton}
                onClick={() => {
                  selectRelativePage(-1)
                }}
                aria-label={translate({
                  id: 'homepage.videos.previousPage.aria',
                  message: 'Show previous page of videos',
                })}
              >
                <span aria-hidden="true">←</span>
                <Translate id="homepage.videos.previousPage">Previous page</Translate>
              </button>
              <span className={styles.videoCount} aria-live="polite">
                {activePage + 1}
                {' / '}
                {pageCount}
              </span>
              <button
                type="button"
                className={styles.directionButton}
                onClick={() => {
                  selectRelativePage(1)
                }}
                aria-label={translate({
                  id: 'homepage.videos.nextPage.aria',
                  message: 'Show next page of videos',
                })}
              >
                <Translate id="homepage.videos.nextPage">Next page</Translate>
                <span aria-hidden="true">→</span>
              </button>
            </nav>
          )}
        </div>
      </div>
    </section>
  )
}
