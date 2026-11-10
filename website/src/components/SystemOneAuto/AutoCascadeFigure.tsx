import React from 'react'
import useBaseUrl from '@docusaurus/useBaseUrl'

const ink = '#10273B'
const blue = '#087CC5'
const amber = '#C68800'
const muted = '#526779'

function DocumentIcon({ x, answer = false }: { x: number, answer?: boolean }) {
  return (
    <g transform={`translate(${x} 451)`} aria-hidden="true">
      <path d="M10 0h57l33 33v87a10 10 0 0 1-10 10H10a10 10 0 0 1-10-10V10A10 10 0 0 1 10 0Z" fill="#FFF" stroke={ink} strokeWidth="4" strokeLinejoin="round" />
      <path d="M67 1v32h32" fill="none" stroke={ink} strokeWidth="4" strokeLinejoin="round" />
      {answer
        ? <path d="m24 67 17 17 34-37" fill="none" stroke={blue} strokeWidth="7" strokeLinecap="round" strokeLinejoin="round" />
        : <path d="M32 56c0-18 34-18 34 0 0 12-17 11-17 28m0 16v2" fill="none" stroke={blue} strokeWidth="7" strokeLinecap="round" />}
      <path d="M24 113h51" stroke="#AEBECC" strokeWidth="4" strokeLinecap="round" />
    </g>
  )
}

function ModelChip({ x, name, size, warm = false }: { x: number, name: string, size: string, warm?: boolean }) {
  const accent = warm ? amber : blue
  return (
    <g transform={`translate(${x} 421)`}>
      <path d="M48-18V0M120-18V0M192-18V0M48 190v18M120 190v18M192 190v18M-18 48H0M-18 95H0M-18 142H0M240 48h18M240 95h18M240 142h18" fill="none" stroke={ink} strokeWidth="4" strokeLinecap="round" />
      <rect width="240" height="190" rx="20" fill={warm ? '#FFF8E4' : '#EEF8FF'} stroke={ink} strokeWidth="4" />
      <rect x="11" y="11" width="218" height="168" rx="13" fill="none" stroke={accent} strokeWidth="2.5" />
      <text x="120" y="83" textAnchor="middle" fontSize="64" fontWeight="700" fill={ink}>{name}</text>
      <text x="120" y="153" textAnchor="middle" fontSize="44" fill={accent}>{size}</text>
    </g>
  )
}

export function AutoCascadeFigure() {
  return (
    <svg
      viewBox="0 0 1760 880"
      width="1760"
      height="880"
      role="img"
      aria-label="The Kai to Vega experiment: Kai answers the original questions. Accept Kai's complete native answer when it passes the configured gate; otherwise ask Vega the same original questions. Both paths return one native answer."
      fontFamily="Inter, Arial, sans-serif"
      fontSize="34"
      fill={ink}
    >
      <rect width="1760" height="880" fill="#FFF" />
      <image href={useBaseUrl('/img/vllm-sr-logo.light.png')} x="80" y="52" width="240" height="79" />
      <text x="1680" y="107" textAnchor="end" fontSize="32" fontWeight="600" fill={muted}>THE KAI → VEGA EXPERIMENT</text>

      <text x="80" y="228" fontSize="68" fontWeight="700" letterSpacing="-2">
        Small first.
        <tspan fill={blue}> Bigger when needed.</tspan>
      </text>

      <path d="M900 440V318H1570V423" fill="none" stroke={blue} strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="m1558 409 12 16 12-16" fill="none" stroke={blue} strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
      <text x="1235" y="291" textAnchor="middle" fontSize="38" fontWeight="650" fill={blue}>Accept Kai</text>

      <DocumentIcon x={135} />
      <path d="M280 516h77m-14-12 14 12-14 12" fill="none" stroke={blue} strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
      <ModelChip x={395} name="Kai" size="0.6B" />
      <path d="M685 516h104m-14-12 14 12-14 12" fill="none" stroke={blue} strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />

      <path d="m900 440 76 76-76 76-76-76 76-76Z" fill="#EEF8FF" stroke={ink} strokeWidth="4" strokeLinejoin="round" />
      <text x="900" y="540" textAnchor="middle" fontSize="68" fontWeight="600" fill={blue}>?</text>

      <path d="M1003 516h69m-14-12 14 12-14 12" fill="none" stroke={amber} strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
      <text x="1230" y="379" textAnchor="middle" fontSize="36" fontWeight="650" fill={amber}>Upgrade to Vega</text>
      <ModelChip x={1110} name="Vega" size="27B" warm />
      <path d="M1395 516h90m-14-12 14 12-14 12" fill="none" stroke={amber} strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
      <DocumentIcon x={1520} answer />

      <g textAnchor="middle" fontSize="34" fill={muted}>
        <text x="185" y="696">Your questions</text>
        <text x="515" y="696">Answers first</text>
        <text x="900" y="696">Check answer</text>
        <text x="1230" y="696">Same questions</text>
        <text x="1570" y="696">Native answer</text>
      </g>
    </svg>
  )
}
