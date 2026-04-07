import React from 'react'

export type CapabilityGlyphKind =
  | 'signal'
  | 'projection'
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
  strokeWidth: 1.7,
}

function SignalGlyph(): JSX.Element {
  return (
    <>
      <path d="M16 28h20M16 40h24M16 52h20" {...strokeProps} opacity="0.62" />
      <path d="M36 28c6 0 8 0 12 4M40 40h8M36 52c6 0 8 0 12-4" {...strokeProps} />
      <rect x="48" y="20" width="26" height="40" rx="11" {...strokeProps} />
      <path d="M61 28v0m0 12v0m0 12v0" {...strokeProps} />
      <circle cx="61" cy="28" r="3.4" fill="currentColor" opacity="0.18" />
      <circle cx="61" cy="40" r="3.4" fill="currentColor" opacity="0.22" />
      <circle cx="61" cy="52" r="3.4" fill="currentColor" opacity="0.18" />
      <path d="M74 40h10" {...strokeProps} />
      <rect x="86" y="22" width="42" height="36" rx="10" {...strokeProps} opacity="0.88" />
      <path d="M92 30h30" {...strokeProps} opacity="0.36" />
      <circle cx="98" cy="40" r="2.8" fill="currentColor" opacity="0.18" />
      <circle cx="107" cy="40" r="2.8" fill="currentColor" opacity="0.26" />
      <circle cx="116" cy="40" r="2.8" fill="currentColor" opacity="0.34" />
      <circle cx="98" cy="48" r="2.8" fill="currentColor" opacity="0.12" />
      <circle cx="107" cy="48" r="2.8" fill="currentColor" opacity="0.2" />
      <circle cx="116" cy="48" r="2.8" fill="currentColor" opacity="0.28" />
    </>
  )
}

function DecisionGlyph(): JSX.Element {
  return (
    <>
      <circle cx="24" cy="30" r="5.5" {...strokeProps} />
      <circle cx="24" cy="50" r="5.5" {...strokeProps} />
      <circle cx="42" cy="40" r="5.5" {...strokeProps} />
      <path d="M29 30l9 7m-9 20l9-7m9-10h13" {...strokeProps} opacity="0.88" />
      <path d="M72 28l12 12-12 12-12-12z" {...strokeProps} />
      <circle cx="72" cy="40" r="2.8" fill="currentColor" opacity="0.2" />
      <rect x="96" y="22" width="24" height="36" rx="8" {...strokeProps} opacity="0.88" />
      <path d="M102 32h12m-12 8h12m-12 8h8" {...strokeProps} opacity="0.56" />
      <path d="M84 40h12" {...strokeProps} />
      <path d="M72 52v8l-18 10m18-10l18 10" {...strokeProps} />
      <rect x="40" y="70" width="24" height="10" rx="5" {...strokeProps} opacity="0.55" />
      <rect x="80" y="70" width="24" height="10" rx="5" {...strokeProps} />
      <circle cx="92" cy="75" r="2.6" fill="currentColor" opacity="0.24" />
    </>
  )
}

function ProjectionGlyph(): JSX.Element {
  return (
    <>
      <circle cx="22" cy="28" r="4.5" {...strokeProps} opacity="0.72" />
      <circle cx="22" cy="48" r="4.5" {...strokeProps} />
      <circle cx="22" cy="68" r="4.5" {...strokeProps} opacity="0.72" />
      <path d="M27 28h16m-16 20h16m-16 20h16" {...strokeProps} opacity="0.72" />
      <rect x="48" y="20" width="28" height="56" rx="10" {...strokeProps} />
      <path d="M56 32h12m-12 12h12m-12 12h12" {...strokeProps} opacity="0.52" />
      <circle cx="62" cy="60" r="3" fill="currentColor" opacity="0.2" />
      <path d="M76 48h14" {...strokeProps} />
      <path d="M90 48l10-12m-10 12 10 12" {...strokeProps} />
      <rect x="102" y="24" width="20" height="12" rx="6" {...strokeProps} opacity="0.82" />
      <rect x="102" y="42" width="24" height="12" rx="6" {...strokeProps} />
      <rect x="102" y="60" width="18" height="12" rx="6" {...strokeProps} opacity="0.82" />
    </>
  )
}

function PluginGlyph(): JSX.Element {
  return (
    <>
      <path d="M20 48h88" {...strokeProps} />
      <path d="M108 48h14" {...strokeProps} />
      <path d="M116 40l8 8-8 8" {...strokeProps} />
      <rect x="24" y="20" width="22" height="16" rx="7" {...strokeProps} />
      <path d="M35 36v12" {...strokeProps} />
      <path d="M31 28h8" {...strokeProps} opacity="0.52" />
      <rect x="58" y="56" width="24" height="16" rx="7" {...strokeProps} opacity="0.92" />
      <path d="M70 48v8" {...strokeProps} />
      <path d="M64 64h12" {...strokeProps} opacity="0.52" />
      <rect x="94" y="18" width="20" height="18" rx="7" {...strokeProps} />
      <path d="M104 36v12" {...strokeProps} />
      <circle cx="35" cy="28" r="2.8" fill="currentColor" opacity="0.18" />
      <circle cx="70" cy="64" r="2.8" fill="currentColor" opacity="0.22" />
      <circle cx="104" cy="27" r="2.8" fill="currentColor" opacity="0.18" />
    </>
  )
}

function LanguageGlyph(): JSX.Element {
  return (
    <>
      <path d="M18 30a10 10 0 0 1 10-10h18a10 10 0 0 1 10 10v12a10 10 0 0 1-10 10H34l-8 8v-8h2a10 10 0 0 1-10-10z" {...strokeProps} />
      <path d="M28 30h18m-18 8h14" {...strokeProps} opacity="0.58" />
      <path d="M56 36h16" {...strokeProps} />
      <path d="M64 28l10 8-10 8" {...strokeProps} />
      <circle cx="96" cy="28" r="6" {...strokeProps} />
      <path d="M96 34v10m0 0L80 56m16-12l16 12" {...strokeProps} />
      <rect x="68" y="58" width="22" height="12" rx="6" {...strokeProps} opacity="0.82" />
      <rect x="102" y="58" width="22" height="12" rx="6" {...strokeProps} />
      <circle cx="96" cy="28" r="2.4" fill="currentColor" opacity="0.22" />
    </>
  )
}

function EconomicsGlyph(): JSX.Element {
  return (
    <>
      <path d="M18 26h20m-20 22h28m-20 22h20" {...strokeProps} opacity="0.52" />
      <path d="M38 26c16 0 24 7 30 22m-22 0h22m-30 22c14 0 24-9 30-22" {...strokeProps} opacity="0.72" />
      <circle cx="72" cy="48" r="11" {...strokeProps} />
      <circle cx="72" cy="48" r="3" fill="currentColor" opacity="0.24" />
      <path d="M83 48h10" {...strokeProps} />
      <path d="M93 48c7-6 13-11 21-14m-21 14h26m-26 0c7 6 13 11 21 14" {...strokeProps} />
      <rect x="116" y="29" width="10" height="10" rx="5" {...strokeProps} opacity="0.62" />
      <rect x="120" y="43" width="14" height="10" rx="5" {...strokeProps} />
      <rect x="116" y="57" width="10" height="10" rx="5" {...strokeProps} opacity="0.62" />
      <path d="M68 31l4-8 4 8" {...strokeProps} opacity="0.4" />
    </>
  )
}

function SafetyGlyph(): JSX.Element {
  return (
    <>
      <path d="M18 48h18m-14-14h10m-10 28h10" {...strokeProps} opacity="0.46" />
      <rect x="42" y="24" width="24" height="48" rx="12" {...strokeProps} />
      <path d="M54 31v34" {...strokeProps} opacity="0.58" />
      <circle cx="54" cy="48" r="4" fill="currentColor" opacity="0.18" />
      <path d="M66 48h10" {...strokeProps} />
      <path d="M88 24l15 6v16c0 13-6 21-15 27-9-6-15-14-15-27V30l15-6z" {...strokeProps} />
      <path d="M81 48l5 5 10-12" {...strokeProps} />
      <rect x="108" y="38" width="16" height="20" rx="6" {...strokeProps} opacity="0.82" />
      <path d="M116 42v12m-5-6h10" {...strokeProps} opacity="0.46" />
      <circle cx="88" cy="36" r="2.4" fill="currentColor" opacity="0.22" />
    </>
  )
}

function MeshGlyph(): JSX.Element {
  return (
    <>
      <rect x="18" y="36" width="20" height="16" rx="6" {...strokeProps} opacity="0.82" />
      <path d="M24 58v8h8" {...strokeProps} opacity="0.42" />
      <circle cx="58" cy="28" r="5" {...strokeProps} opacity="0.72" />
      <circle cx="54" cy="48" r="5" {...strokeProps} />
      <circle cx="58" cy="68" r="5" {...strokeProps} opacity="0.72" />
      <circle cx="82" cy="38" r="5" {...strokeProps} />
      <circle cx="82" cy="58" r="5" {...strokeProps} />
      <path d="M38 44h11m9-12 20 8m-24 8h23m-19 16 20-8m4-18v16" {...strokeProps} />
      <path d="M88 48h10" {...strokeProps} />
      <path d="M101 60h21a8 8 0 0 0 0-16 10 10 0 0 0-18-4 7 7 0 0 0-3 20z" {...strokeProps} />
      <circle cx="106" cy="48" r="2.6" fill="currentColor" opacity="0.22" />
      <circle cx="114" cy="48" r="2.6" fill="currentColor" opacity="0.3" />
      <circle cx="122" cy="48" r="2.6" fill="currentColor" opacity="0.18" />
    </>
  )
}

function SelectionGlyph(): JSX.Element {
  return (
    <>
      <rect x="18" y="24" width="30" height="10" rx="5" {...strokeProps} opacity="0.52" />
      <rect x="18" y="44" width="40" height="10" rx="5" {...strokeProps} />
      <rect x="18" y="64" width="26" height="10" rx="5" {...strokeProps} opacity="0.52" />
      <path d="M48 29c10 0 14 4 20 16m-10 4h10m-24 20c12 0 18-10 24-20" {...strokeProps} opacity="0.52" />
      <circle cx="70" cy="49" r="6" {...strokeProps} />
      <circle cx="70" cy="49" r="2.5" fill="currentColor" opacity="0.2" />
      <path d="M76 47c14-2 24-12 34-20" {...strokeProps} />
      <path d="M76 53c16 2 26 10 34 16" {...strokeProps} opacity="0.52" />
      <path
        d="M114 24l2.7 5.8 6.4.9-4.6 4.5 1.1 6.4-5.6-3-5.6 3 1.1-6.4-4.6-4.5 6.4-.9z"
        {...strokeProps}
      />
      <circle cx="114" cy="69" r="10" {...strokeProps} opacity="0.82" />
      <circle cx="114" cy="69" r="2.4" fill="currentColor" opacity="0.22" />
    </>
  )
}

function DocsGlyph(): JSX.Element {
  return (
    <>
      <rect x="18" y="18" width="108" height="60" rx="11" {...strokeProps} />
      <path d="M18 32h108" {...strokeProps} opacity="0.52" />
      <circle cx="28" cy="25" r="1.9" fill="currentColor" opacity="0.22" />
      <circle cx="34" cy="25" r="1.9" fill="currentColor" opacity="0.18" />
      <circle cx="40" cy="25" r="1.9" fill="currentColor" opacity="0.14" />
      <rect x="28" y="40" width="18" height="26" rx="5" {...strokeProps} opacity="0.7" />
      <path d="M34 46v12m6-8v8" {...strokeProps} opacity="0.52" />
      <rect x="54" y="40" width="40" height="26" rx="6" {...strokeProps} />
      <path d="M60 47h24" {...strokeProps} opacity="0.34" />
      <path d="M60 58l8-8 8 4 8-10 6 4" {...strokeProps} />
      <circle cx="68" cy="50" r="2.2" fill="currentColor" opacity="0.18" />
      <circle cx="76" cy="54" r="2.2" fill="currentColor" opacity="0.2" />
      <circle cx="84" cy="44" r="2.2" fill="currentColor" opacity="0.22" />
      <rect x="102" y="40" width="16" height="10" rx="5" {...strokeProps} opacity="0.82" />
      <rect x="102" y="56" width="16" height="10" rx="5" {...strokeProps} opacity="0.64" />
      <path d="M105 45h10m-10 16h7" {...strokeProps} opacity="0.46" />
    </>
  )
}

function renderGlyph(kind: CapabilityGlyphKind): JSX.Element {
  switch (kind) {
    case 'signal':
      return <SignalGlyph />
    case 'projection':
      return <ProjectionGlyph />
    case 'decision':
      return <DecisionGlyph />
    case 'plugin':
      return <PluginGlyph />
    case 'language':
      return <LanguageGlyph />
    case 'selection':
      return <SelectionGlyph />
    case 'economics':
      return <EconomicsGlyph />
    case 'safety':
      return <SafetyGlyph />
    case 'mesh':
      return <MeshGlyph />
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
