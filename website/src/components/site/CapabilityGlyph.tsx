import React from 'react'

export type CapabilityGlyphKind =
  | 'signal'
  | 'decision'
  | 'plugin'
  | 'language'
  | 'selection'
  | 'docs'

type CapabilityGlyphProps = {
  className?: string
  kind: CapabilityGlyphKind
}

const strokeProps = {
  fill: 'none',
  stroke: 'currentColor',
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  strokeWidth: 1.6,
}

function SignalGlyph(): JSX.Element {
  return (
    <>
      <circle cx="28" cy="22" r="7" {...strokeProps} />
      <circle cx="72" cy="18" r="7" {...strokeProps} />
      <circle cx="116" cy="22" r="7" {...strokeProps} />
      <path d="M28 29v15m44-19v19m44-15v15" {...strokeProps} opacity="0.82" />
      <path d="M28 44c12 12 28 18 44 18s32-6 44-18" {...strokeProps} />
      <path d="M52 70h40" {...strokeProps} />
      <path d="M44 78h56" {...strokeProps} opacity="0.6" />
      <circle cx="28" cy="22" r="2.2" fill="currentColor" opacity="0.18" />
      <circle cx="72" cy="18" r="2.2" fill="currentColor" opacity="0.18" />
      <circle cx="116" cy="22" r="2.2" fill="currentColor" opacity="0.18" />
    </>
  )
}

function DecisionGlyph(): JSX.Element {
  return (
    <>
      <circle cx="72" cy="30" r="12" {...strokeProps} />
      <path d="M72 42v12" {...strokeProps} />
      <path d="M72 54H34m38 0h38" {...strokeProps} />
      <rect x="20" y="58" width="28" height="18" rx="9" {...strokeProps} />
      <rect x="58" y="58" width="28" height="18" rx="9" {...strokeProps} />
      <rect x="96" y="58" width="28" height="18" rx="9" {...strokeProps} />
      <path d="M66 30h12" {...strokeProps} />
      <path d="M72 24v12" {...strokeProps} />
      <circle cx="72" cy="30" r="3" fill="currentColor" opacity="0.18" />
    </>
  )
}

function PluginGlyph(): JSX.Element {
  return (
    <>
      <rect x="16" y="28" width="30" height="40" rx="8" {...strokeProps} />
      <rect x="57" y="20" width="30" height="56" rx="8" {...strokeProps} />
      <rect x="98" y="28" width="30" height="40" rx="8" {...strokeProps} />
      <path d="M46 48h11m30 0h11" {...strokeProps} />
      <path d="M32 36v24m40-32v40m40-32v24" {...strokeProps} opacity="0.55" />
      <circle cx="32" cy="48" r="4" fill="currentColor" opacity="0.18" />
      <circle cx="72" cy="48" r="4" fill="currentColor" opacity="0.18" />
      <circle cx="112" cy="48" r="4" fill="currentColor" opacity="0.18" />
    </>
  )
}

function LanguageGlyph(): JSX.Element {
  return (
    <>
      <circle cx="34" cy="48" r="12" {...strokeProps} />
      <path d="M46 48h20" {...strokeProps} />
      <path d="M66 48l9-9m-9 9l9 9" {...strokeProps} />
      <path d="M82 30h28v12H82zm0 24h28v12H82z" {...strokeProps} />
      <path d="M96 42v12" {...strokeProps} opacity="0.72" />
      <path d="M74 48h8m28 0h8" {...strokeProps} opacity="0.78" />
      <circle cx="34" cy="48" r="3.2" fill="currentColor" opacity="0.18" />
      <circle cx="96" cy="36" r="2.8" fill="currentColor" opacity="0.18" />
      <circle cx="96" cy="60" r="2.8" fill="currentColor" opacity="0.18" />
    </>
  )
}

function SelectionGlyph(): JSX.Element {
  return (
    <>
      <path d="M20 26h78" {...strokeProps} opacity="0.45" />
      <path d="M20 46h96" {...strokeProps} />
      <path d="M20 66h62" {...strokeProps} opacity="0.62" />
      <rect x="20" y="20" width="78" height="12" rx="6" {...strokeProps} />
      <rect x="20" y="40" width="96" height="12" rx="6" {...strokeProps} />
      <rect x="20" y="60" width="62" height="12" rx="6" {...strokeProps} />
      <path d="M108 46h18" {...strokeProps} />
      <path d="M118 38l8 8-8 8" {...strokeProps} />
      <circle cx="30" cy="46" r="3.2" fill="currentColor" opacity="0.18" />
    </>
  )
}

function DocsGlyph(): JSX.Element {
  return (
    <>
      <rect x="20" y="18" width="34" height="48" rx="8" {...strokeProps} />
      <rect x="55" y="28" width="34" height="48" rx="8" {...strokeProps} opacity="0.82" />
      <rect x="90" y="18" width="34" height="48" rx="8" {...strokeProps} />
      <path d="M32 32h10m-10 12h10m35-4h10" {...strokeProps} opacity="0.68" />
      <path d="M54 42h9m26 0h9" {...strokeProps} />
      <circle cx="37" cy="54" r="3.2" fill="currentColor" opacity="0.18" />
      <circle cx="72" cy="54" r="3.2" fill="currentColor" opacity="0.18" />
      <circle cx="107" cy="54" r="3.2" fill="currentColor" opacity="0.18" />
    </>
  )
}

function renderGlyph(kind: CapabilityGlyphKind): JSX.Element {
  switch (kind) {
    case 'signal':
      return <SignalGlyph />
    case 'decision':
      return <DecisionGlyph />
    case 'plugin':
      return <PluginGlyph />
    case 'language':
      return <LanguageGlyph />
    case 'selection':
      return <SelectionGlyph />
    case 'docs':
      return <DocsGlyph />
    default:
      return <SignalGlyph />
  }
}

export default function CapabilityGlyph({
  className,
  kind,
}: CapabilityGlyphProps): JSX.Element {
  return (
    <svg
      aria-hidden="true"
      className={className}
      viewBox="0 0 144 96"
      xmlns="http://www.w3.org/2000/svg"
    >
      {renderGlyph(kind)}
    </svg>
  )
}
