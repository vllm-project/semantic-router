import { useEffect, useState } from 'react'

import ServiceNotConfigured, { type ServiceConfig } from './ServiceNotConfigured'
import ProductLoadingState from './ProductLoadingState'
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
  const [availability, setAvailability] = useState<'checking' | 'available' | 'missing' | 'failed'>(
    'checking',
  )
  const [frameKey, setFrameKey] = useState(0)
  const [frameLoading, setFrameLoading] = useState(true)
  const [frameSlow, setFrameSlow] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    let timedOut = false
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      timedOut = true
      controller.abort()
    }, 12000)

    const run = async () => {
      try {
        let response = await fetch(availabilityUrl, {
          method: 'HEAD',
          signal: controller.signal,
        })
        if (response.status === 405) {
          response = await fetch(availabilityUrl, { signal: controller.signal })
        }
        if (!active) return
        if (response.status === 503) {
          setAvailability('missing')
        } else if (response.ok) {
          setAvailability('available')
        } else {
          setAvailability('failed')
          setFrameLoading(false)
          setError(
            response.status === 401 || response.status === 403
              ? `Access to ${service.name} was denied. Check your session and permissions, then try again.`
              : `${service.name} is unavailable (HTTP ${response.status}). Try again when the service is ready.`,
          )
        }
      } catch (requestError) {
        if (
          active &&
          (timedOut ||
            !(requestError instanceof DOMException && requestError.name === 'AbortError'))
        ) {
          setAvailability('failed')
          setFrameLoading(false)
          setError(
            timedOut
              ? `${service.name} took too long to respond. Try again.`
              : `Could not connect to ${service.name}. Check your connection and try again.`,
          )
        }
      } finally {
        window.clearTimeout(timer)
      }
    }

    void run()
    return () => {
      active = false
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [availabilityUrl, frameKey, service.name])

  useEffect(() => {
    if (!frameLoading || availability !== 'available') return
    const timer = window.setTimeout(() => setFrameSlow(true), 8000)
    return () => window.clearTimeout(timer)
  }, [availability, frameKey, frameLoading])

  const reloadFrame = () => {
    setAvailability('checking')
    setFrameLoading(true)
    setFrameSlow(false)
    setError(null)
    setFrameKey((value) => value + 1)
  }

  if (availability === 'missing') {
    return (
      <div className={styles.page}>
        <ServiceNotConfigured service={service} onRetry={reloadFrame} />
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
            {availability === 'checking'
              ? 'Checking connection'
              : availability === 'failed'
                ? 'Connection unavailable'
                : 'Connected through dashboard'}
          </span>
          <button type="button" onClick={reloadFrame} disabled={availability === 'checking'}>
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

      {availability !== 'failed' ? (
        <section className={styles.frameShell} aria-label={`${title} embedded workspace`}>
          <div className={styles.frameRail}>
            <span>{service.name}</span>
            <span>Secure same-origin proxy</span>
          </div>

          {(availability === 'checking' || frameLoading) && (
            <div className={styles.loading} role="status" aria-live="polite">
              <ProductLoadingState
                label={
                  frameSlow ? `Still connecting to ${service.name}` : `Loading ${service.name}`
                }
                compact
              />
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
                setAvailability('failed')
                setFrameLoading(false)
                setFrameSlow(false)
                setError(
                  `Could not load ${service.name}. Check the service and proxy configuration.`,
                )
              }}
            />
          ) : null}
        </section>
      ) : null}
    </div>
  )
}
