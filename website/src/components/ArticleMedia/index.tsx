import React, { useId, useState } from 'react'
import useBaseUrl from '@docusaurus/useBaseUrl'
import styles from './styles.module.css'

type VideoProps = {
  src: string
  poster: string
  title: string
  children: React.ReactNode
}

export function ArticleVideo({ src, poster, title, children }: VideoProps) {
  const videoUrl = useBaseUrl(src)
  const posterUrl = useBaseUrl(poster)

  return (
    <figure className={styles.videoFigure}>
      <video
        className={styles.video}
        controls
        playsInline
        preload="metadata"
        poster={posterUrl}
        aria-label={title}
        width="1080"
        height="1080"
      >
        <source src={videoUrl} type="video/mp4" />
        <a href={videoUrl}>{title}</a>
      </video>
      <figcaption className={styles.caption}>{children}</figcaption>
    </figure>
  )
}

type FigureProps = {
  src: string
  alt: string
  width: number
  height: number
  children?: React.ReactNode
}

export function ArticleFigure({ src, alt, width, height, children }: FigureProps) {
  const imageUrl = useBaseUrl(src)

  return (
    <figure className={styles.figure}>
      <a href={imageUrl} target="_blank" rel="noopener noreferrer" className={styles.imageLink}>
        <img
          className={styles.image}
          src={imageUrl}
          alt={alt}
          width={width}
          height={height}
          loading="lazy"
          decoding="async"
        />
      </a>
      {children && <figcaption className={styles.caption}>{children}</figcaption>}
    </figure>
  )
}

type Metric = {
  label: string
  value: string
  before: string
  measure: string
  source: string
}

export function ArticleMetrics({ items }: { items: Metric[] }) {
  return (
    <dl className={styles.metrics}>
      {items.map(item => (
        <div className={styles.metric} key={item.label}>
          <dt><a href={item.source}>{item.label}</a></dt>
          <dd>
            <strong>{item.value}</strong>
            <span>{item.measure}</span>
            <small>{`from ${item.before} · previous mmBERT`}</small>
          </dd>
        </div>
      ))}
    </dl>
  )
}

type Chart = FigureProps & { label: string }

export function ArticleChartGallery({ charts }: { charts: Chart[] }) {
  const [selected, setSelected] = useState(0)
  const panelId = useId()
  const chart = charts[selected]

  return (
    <div className={styles.chartGallery}>
      <div className={styles.chartChoices} role="group" aria-label="Choose an Omni benchmark">
        {charts.map((item, index) => (
          <button
            type="button"
            key={item.label}
            aria-pressed={selected === index}
            aria-controls={panelId}
            onClick={() => setSelected(index)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div id={panelId} aria-live="polite">
        <ArticleFigure {...chart} />
      </div>
    </div>
  )
}
