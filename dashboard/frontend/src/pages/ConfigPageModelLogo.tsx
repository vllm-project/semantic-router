import type { BuiltInModelCatalog, CatalogPresentation } from '../types/modelCatalog'
import { LogoMark, ProviderLogo, cn } from '../ui'
import { bundledCatalog, catalogProvider, modelProviderIds } from './configPageModelPresentation'
import { resolveModelCatalogIcon } from './modelProviderIcons'
import type { NormalizedModel } from './configPageSupport'

function PresentationLogo({
  presentation,
  fallback,
  size,
  framed,
}: {
  presentation?: CatalogPresentation
  fallback: string
  size: number
  framed: boolean
}) {
  if (presentation?.logo.startsWith('package:')) {
    return (
      <ProviderLogo
        provider={presentation.logo.slice('package:'.length)}
        size={size}
        framed={framed}
      />
    )
  }
  const src = presentation ? resolveModelCatalogIcon(presentation.logo) : ''
  const glyph = src ? (
    <img
      src={src}
      alt=""
      width={size}
      height={size}
      className={cn('shrink-0 object-contain', presentation?.monochrome && 'dark:invert')}
      style={{ width: size, height: size }}
    />
  ) : (
    <span
      className="grid shrink-0 place-items-center rounded-[6px] bg-muted font-semibold uppercase text-muted-foreground"
      style={{ width: size, height: size, fontSize: Math.max(9, size * 0.45) }}
    >
      {(presentation?.monogram || fallback).slice(0, 2)}
    </span>
  )
  if (!framed) return <span className="inline-flex">{glyph}</span>
  return (
    <span
      className="grid shrink-0 place-items-center rounded-lg border border-border bg-surface shadow-xs"
      style={{ width: size + 14, height: size + 14 }}
    >
      {glyph}
    </span>
  )
}

/** The logo of a catalog provider, such as `vllm` or `openai-compatible`. */
export function CatalogProviderLogo({
  provider,
  size = 18,
  framed = false,
  catalog = bundledCatalog,
}: {
  provider: string
  size?: number
  framed?: boolean
  catalog?: BuiltInModelCatalog
}) {
  const entry = catalogProvider(provider, catalog)
  if (!entry) return <ProviderLogo provider={provider} size={size} framed={framed} />
  return (
    <span title={entry.display_name} className="inline-flex shrink-0">
      <PresentationLogo
        presentation={entry.presentation}
        fallback={provider}
        size={size}
        framed={framed}
      />
    </span>
  )
}

/** A backend model's logo: its first provider, else its catalog card, else the Router mark. */
export function ModelLogo({
  model,
  size = 18,
  framed = false,
  catalog = bundledCatalog,
}: {
  model: Pick<NormalizedModel, 'name' | 'catalog' | 'backend_refs'>
  size?: number
  framed?: boolean
  catalog?: BuiltInModelCatalog
}) {
  const provider = modelProviderIds(model)[0]
  if (provider) {
    return <CatalogProviderLogo provider={provider} size={size} framed={framed} catalog={catalog} />
  }
  const card = model.catalog ? catalog.models.find((entry) => entry.id === model.catalog) : null
  if (card) {
    return (
      <span title={card.publisher} className="inline-flex shrink-0">
        <PresentationLogo
          presentation={card.presentation}
          fallback={model.name}
          size={size}
          framed={framed}
        />
      </span>
    )
  }
  const mark = <LogoMark size={size} />
  if (!framed) return <span className="inline-flex">{mark}</span>
  return (
    <span
      className="grid shrink-0 place-items-center rounded-lg border border-border bg-surface shadow-xs"
      style={{ width: size + 14, height: size + 14 }}
    >
      {mark}
    </span>
  )
}
