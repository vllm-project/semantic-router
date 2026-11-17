import { useEffect, useState } from 'react'
import { AppWindow, ExternalLink, RotateCw, ShieldCheck, type LucideIcon } from 'lucide-react'

import {
  Alert,
  Button,
  IconTile,
  PageContainer,
  PageHeader,
  StatusLabel,
  type StatusTone,
  type Tone,
} from '../ui'
import ProductLoadingState from './ProductLoadingState'
import ServiceNotConfigured, { type ServiceConfig } from './ServiceNotConfigured'

interface EmbeddedServicePageProps {
  /** Retained for API compatibility; page headers no longer render an eyebrow. */
  eyebrow?: string
  title: string
  description: string
  service: ServiceConfig
  availabilityUrl: string
  src: string
  iframeTitle: string
  icon?: LucideIcon
  tone?: Tone
}

type Availability = 'checking' | 'available' | 'missing' | 'failed'

const availabilityStatus: Record<Availability, { tone: StatusTone; label: string }> = {
  checking: { tone: 'info', label: 'Checking connection' },
  available: { tone: 'success', label: 'Connected through dashboard' },
  missing: { tone: 'neutral', label: 'Not configured' },
  failed: { tone: 'danger', label: 'Connection unavailable' },
}

export default function EmbeddedServicePage({
  title,
  description,
  service,
  availabilityUrl,
  src,
  iframeTitle,
  icon = AppWindow,
  tone = 'blue',
}: EmbeddedServicePageProps) {
  const [availability, setAvailability] = useState<Availability>('checking')
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

  const status = availabilityStatus[availability]

  return (
    <PageContainer width="wide">
      <PageHeader
        icon={<IconTile icon={icon} tone={tone} size="lg" />}
        title={title}
        description={description}
        actions={
          availability === 'missing' ? null : (
            <>
              <span className="mr-1" aria-live="polite">
                <StatusLabel tone={status.tone} pulse={availability === 'checking'}>
                  {status.label}
                </StatusLabel>
              </span>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={reloadFrame}
                disabled={availability === 'checking'}
              >
                <RotateCw />
                Reload
              </Button>
              <Button variant="outline" size="sm" asChild>
                <a href={src} target="_blank" rel="noopener noreferrer">
                  <ExternalLink />
                  Open full view
                </a>
              </Button>
            </>
          )
        }
      />

      {availability === 'missing' ? (
        <ServiceNotConfigured service={service} onRetry={reloadFrame} />
      ) : null}

      {error ? (
        <div role="alert" className="mb-4">
          <Alert
            tone="danger"
            title={`${service.name} could not be loaded`}
            actions={
              <Button type="button" variant="outline" size="sm" onClick={reloadFrame}>
                Try again
              </Button>
            }
          >
            {error}
          </Alert>
        </div>
      ) : null}

      {availability !== 'failed' && availability !== 'missing' ? (
        <section
          className="flex flex-col overflow-hidden rounded-xl border border-border bg-card shadow-xs"
          aria-label={`${title} embedded workspace`}
        >
          <div className="flex h-10 items-center justify-between gap-3 border-b border-border bg-surface-subtle/70 px-4 text-xs text-muted-foreground">
            <span className="font-medium text-foreground/85">{service.name}</span>
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="size-3.5 text-success" aria-hidden="true" />
              Secure same-origin proxy
            </span>
          </div>

          <div className="relative h-[calc(100dvh-15rem)] min-h-[520px] bg-background">
            {(availability === 'checking' || frameLoading) && (
              <div className="absolute inset-0 grid place-items-center">
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
                className="relative size-full border-0"
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
          </div>
        </section>
      ) : null}
    </PageContainer>
  )
}
