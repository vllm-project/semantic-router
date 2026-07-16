import { useCallback, useEffect, useState } from 'react'

import ServiceNotConfigured, { type ServiceConfig } from './ServiceNotConfigured'
import styles from './EmbeddedServicePage.module.css'

interface EmbeddedServicePageProps {
  eyebrow: string
  title: string
  description: string
  service: ServiceConfig
  availabilityUrl: string
  src: string
  iframeTitle: string
}

export default function EmbeddedServicePage({
  eyebrow,
  title,
  description,
  service,
  availabilityUrl,
  src,
  iframeTitle,
}: EmbeddedServicePageProps) {
  const [availability, setAvailability] = useState<'checking' | 'available' | 'missing'>('checking')
  const [frameKey, setFrameKey] = useState(0)
  const [frameLoading, setFrameLoading] = useState(true)
  const [frameSlow, setFrameSlow] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const checkAvailability = useCallback(async () => {
    setAvailability('checking')
    setError(null)
    try {
      const response = await fetch(availabilityUrl, {
        method: 'HEAD',
      })
      setAvailability(response.status === 503 ? 'missing' : 'available')
    } catch {
      // The embedded endpoint can reject HEAD while still serving GET. Let the iframe
      // provide the final connection state instead of turning a transient probe into a dead end.
      setAvailability('available')
    }
  }, [availabilityUrl])

  useEffect(() => {
    let active = true
    const controller = new AbortController()

    const run = async () => {
      try {
        const response = await fetch(availabilityUrl, {
          method: 'HEAD',
          signal: controller.signal,
        })
        if (active) setAvailability(response.status === 503 ? 'missing' : 'available')
      } catch (requestError) {
        if (
          active &&
          !(requestError instanceof DOMException && requestError.name === 'AbortError')
        ) {
          setAvailability('available')
        }
      }
    }

    void run()
    return () => {
      active = false
      controller.abort()
    }
  }, [availabilityUrl])

  useEffect(() => {
    if (!frameLoading || availability !== 'available') return
    const timer = window.setTimeout(() => setFrameSlow(true), 8000)
    return () => window.clearTimeout(timer)
  }, [availability, frameKey, frameLoading])

  const reloadFrame = () => {
    setFrameLoading(true)
    setFrameSlow(false)
    setError(null)
    setFrameKey((value) => value + 1)
  }

  if (availability === 'missing') {
    return (
      <div className={styles.page}>
        <ServiceNotConfigured
          service={service}
          onRetry={() => {
            setFrameLoading(true)
            setFrameSlow(false)
            void checkAvailability()
          }}
        />
      </div>
    )
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.heading}>
          <span className={styles.eyebrow}>{eyebrow}</span>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        <div className={styles.actions}>
          <span className={styles.status} aria-live="polite">
            <span className={styles.statusDot} aria-hidden="true" />
            {availability === 'checking' ? 'Checking connection' : 'Connected through dashboard'}
          </span>
          <button type="button" onClick={reloadFrame} disabled={availability !== 'available'}>
            Reload
          </button>
          <a href={src} target="_blank" rel="noopener noreferrer">
            Open full view
          </a>
        </div>
      </header>

      {error ? (
        <div className={styles.error} role="alert">
          <span>{error}</span>
          <button type="button" onClick={reloadFrame}>
            Try again
          </button>
        </div>
      ) : null}

      <section className={styles.frameShell} aria-label={`${title} embedded workspace`}>
        <div className={styles.frameRail}>
          <span>{service.name}</span>
          <span>Secure same-origin proxy</span>
        </div>

        {(availability === 'checking' || frameLoading) && (
          <div className={styles.loading} role="status" aria-live="polite">
            <span className={styles.spinner} aria-hidden="true" />
            <strong>
              {frameSlow ? `Still connecting to ${service.name}` : `Loading ${service.name}`}
            </strong>
            <p>
              {frameSlow
                ? 'The service is taking longer than expected. You can reload or open the full view.'
                : 'Establishing the embedded observability session.'}
            </p>
          </div>
        )}

        {availability === 'available' ? (
          <iframe
            key={`${frameKey}-${src}`}
            src={src}
            className={styles.frame}
            title={iframeTitle}
            allowFullScreen
            referrerPolicy="same-origin"
            onLoad={() => {
              setFrameLoading(false)
              setFrameSlow(false)
              setError(null)
            }}
            onError={() => {
              setFrameLoading(false)
              setFrameSlow(false)
              setError(`Could not load ${service.name}. Check the service and proxy configuration.`)
            }}
          />
        ) : null}
      </section>
    </div>
  )
}
