import { modelProviderIconAssets } from '../../pages/modelProviderIcons'
import { cn } from '../lib/cn'

/** The vLLM Semantic Router mark: the gold and blue strokes of the vLLM "v". */
export function LogoMark({ size = 24, className, animated = false }: { size?: number; className?: string; animated?: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      className={cn('shrink-0', className)}
      role="img"
      aria-label="vLLM Semantic Router"
    >
      <path
        d="M0 4.973h9.324V23L0 4.973z"
        fill="#FDB515"
        className={animated ? 'origin-bottom animate-in fade-in-0 slide-in-from-bottom-2 duration-700' : undefined}
      />
      <path
        d="M13.986 4.351L22.378 0l-6.216 23H9.324l4.662-18.649z"
        fill="#30A2FF"
        className={animated ? 'animate-in fade-in-0 slide-in-from-top-2 duration-700 [animation-delay:120ms] [animation-fill-mode:both]' : undefined}
      />
    </svg>
  )
}

/**
 * The vLLM Semantic Router wordmark: black lettering in light mode, white in
 * dark mode. Collapsed spaces show only the vLLM mark.
 */
export function BrandLockup({ collapsed = false, className, height = 34 }: { collapsed?: boolean; className?: string; height?: number }) {
  if (collapsed) {
    return (
      <span className={cn('grid size-9 shrink-0 place-items-center', className)}>
        <LogoMark size={24} />
      </span>
    )
  }
  return (
    <span className={cn('flex min-w-0 items-center', className)} style={{ height }}>
      <img src="/brand/vllm-sr-logo-black.png" alt="vLLM Semantic Router" className="block h-full w-auto select-none dark:hidden" draggable={false} />
      <img src="/brand/vllm-sr-logo-white.png" alt="vLLM Semantic Router" className="hidden h-full w-auto select-none dark:block" draggable={false} />
    </span>
  )
}

const monochromeProviders = new Set([
  'openai',
  'anthropic',
  'xai',
  'ollama',
  'moonshot',
  'groq',
  'friendli',
  'lmstudio',
  'nebius',
  'vercel',
  'xiaomimimo',
  'zai',
])

export function ProviderLogo({
  provider,
  size = 20,
  className,
  framed = false,
}: {
  provider: string
  size?: number
  className?: string
  framed?: boolean
}) {
  const key = provider.toLowerCase()
  const src = modelProviderIconAssets[key]
  const image = src ? (
    <img
      src={src}
      alt=""
      width={size}
      height={size}
      className={cn('shrink-0 object-contain', monochromeProviders.has(key) && 'dark:invert')}
      style={{ width: size, height: size }}
    />
  ) : (
    <span
      className="grid shrink-0 place-items-center rounded-[6px] bg-muted font-semibold uppercase text-muted-foreground"
      style={{ width: size, height: size, fontSize: Math.max(9, size * 0.45) }}
    >
      {key.slice(0, 1)}
    </span>
  )
  if (!framed) return <span className={cn('inline-flex', className)} title={provider}>{image}</span>
  return (
    <span
      title={provider}
      className={cn('grid shrink-0 place-items-center rounded-lg border border-border bg-surface shadow-xs', className)}
      style={{ width: size + 14, height: size + 14 }}
    >
      {image}
    </span>
  )
}

export function ProviderStack({ providers, size = 18, max = 4 }: { providers: string[]; size?: number; max?: number }) {
  const shown = providers.slice(0, max)
  const extra = providers.length - shown.length
  return (
    <span className="flex items-center">
      {shown.map((provider, index) => (
        <span
          key={`${provider}-${index}`}
          className="grid place-items-center rounded-full border-2 border-card bg-surface"
          style={{ width: size + 8, height: size + 8, marginLeft: index === 0 ? 0 : -6, zIndex: shown.length - index }}
        >
          <ProviderLogo provider={provider} size={size - 2} />
        </span>
      ))}
      {extra > 0 ? (
        <span
          className="grid place-items-center rounded-full border-2 border-card bg-muted text-[10px] font-semibold text-muted-foreground"
          style={{ width: size + 8, height: size + 8, marginLeft: -6 }}
        >
          +{extra}
        </span>
      ) : null}
    </span>
  )
}
