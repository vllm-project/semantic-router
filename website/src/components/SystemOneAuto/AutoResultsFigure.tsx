import React from 'react'
import useBaseUrl from '@docusaurus/useBaseUrl'
import evidence from '../../../static/img/blog/system-one-auto/source/figure-data.json'

const ink = '#10273B'
const muted = '#526879'
const blue = '#087BCB'
const amber = '#E3A000'
const percent = (value: number) => (value * 100).toFixed(2)
const warmPasses = [evidence.passes['1'], evidence.passes['2']]
const mean = (arm: 'auto' | 'direct_vega') =>
  warmPasses.reduce((total, pass) => total + pass.arms[arm].elapsed_ms.mean, 0) / warmPasses.length

export function AutoResultsFigure() {
  const logo = useBaseUrl('/img/vllm-sr-logo.light.png')
  const meanReduction = Math.round(100 * (1 - mean('auto') / mean('direct_vega')))
  const physicalCalls = evidence.passes['0'].arms
  const estimatedCostSavings = 1 - physicalCalls.auto.physical_calls.vega / physicalCalls.direct_vega.physical_calls.vega
  const rows = [
    { name: 'Kai · 0.6B', accuracy: evidence.quality.direct_kai.accuracy, color: '#A1B8C8', y: 434 },
    { name: 'System One Auto', accuracy: evidence.quality.auto.accuracy, color: blue, y: 540 },
    { name: 'Vega · 27B', accuracy: evidence.quality.direct_vega.accuracy, color: amber, y: 646 },
  ]

  return (
    <svg
      viewBox="0 0 1600 930"
      width="1600"
      height="930"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label={`System One Auto gains ${percent(evidence.paired_quality.direct_kai.accuracy_gain)} percentage points over Kai. Estimated inference cost savings are ${percent(estimatedCostSavings)} percent versus Vega-only, assuming equal cost per Vega call and negligible Kai cost. Warm mean latency is about ${meanReduction} percent lower than direct Vega, while p95 is higher.`}
      fill={ink}
      style={{ background: '#FFFFFF', color: ink, fontFamily: 'Inter, system-ui, sans-serif' }}
    >
      <title>More accurate answers. Selective upgrades.</title>
      <desc>
        {`Historical public JevBench suite: ${evidence.quality_items} items. Kai accuracy ${percent(evidence.quality.direct_kai.accuracy)}%, Auto ${percent(evidence.quality.auto.accuracy)}%, Vega ${percent(evidence.quality.direct_vega.accuracy)}%. All bars start at zero on the same 0–100% scale. The cascade makes ${physicalCalls.auto.physical_calls.vega} Vega calls versus ${physicalCalls.direct_vega.physical_calls.vega} for Vega-only, implying ${percent(estimatedCostSavings)}% estimated inference cost savings only under equal per-Vega-call cost and negligible Kai cost. Two warm passes at concurrency 1 show lower mean latency and higher p95 than Vega. This is not an official v1.6.1 score.`}
      </desc>
      <rect width="1600" height="930" fill="#FFFFFF" />
      <image href={logo} x="70" y="50" width="214" height="70" preserveAspectRatio="xMinYMid meet" />
      <text x="340" y="96" fontSize="32" fontWeight="650" letterSpacing="1" fill={muted}>SYSTEM ONE AUTO · MEASURED RESULTS</text>

      <text x="70" y="232" fontSize="112" fontWeight="800" letterSpacing="-5" fill={blue}>
        {`+${percent(evidence.paired_quality.direct_kai.accuracy_gain)} pp`}
      </text>
      <text x="70" y="297" fontSize="34" fill={ink}>accuracy over Kai alone</text>
      <text x="780" y="212" fontSize="56" fontWeight="750" letterSpacing="-1.5" fill={ink}>More accurate answers.</text>
      <text x="780" y="294" fontSize="56" fontWeight="750" letterSpacing="-1.5" fill={ink}>Selective upgrades.</text>

      <text x="70" y="362" fontSize="32" fontWeight="650" fill={muted}>Exact-label accuracy</text>
      <line x1="385" y1="397" x2="385" y2="680" stroke="#C8D7E1" strokeWidth="2" />
      {rows.map(row => (
        <g key={row.name}>
          <text x="70" y={row.y + 12} fontSize="32" fontWeight={row.name === 'System One Auto' ? '750' : '550'} fill={ink}>{row.name}</text>
          <rect x="385" y={row.y - 26} width="560" height="52" rx="4" fill="#EFF3F6" />
          <rect x="385" y={row.y - 26} width={560 * row.accuracy} height="52" rx="4" fill={row.color} data-accuracy={row.accuracy} data-model={row.name} />
          <text x="1100" y={row.y + 14} textAnchor="end" fontSize="40" fontWeight="750" fill={row.name === 'System One Auto' ? blue : ink}>
            {`${percent(row.accuracy)}%`}
          </text>
        </g>
      ))}
      <text x="385" y="725" textAnchor="middle" fontSize="32" fill={muted}>0%</text>
      <text x="945" y="725" textAnchor="middle" fontSize="32" fill={muted}>100%</text>

      <line x1="1150" y1="375" x2="1150" y2="756" stroke="#DCE6ED" strokeWidth="2" />
      <text x="1210" y="423" fontSize="78" fontWeight="800" letterSpacing="-3" fill={blue}>
        {`${percent(estimatedCostSavings)}%`}
      </text>
      <text x="1210" y="477" fontSize="32" fill={ink}>estimated inference</text>
      <text x="1210" y="521" fontSize="32" fill={ink}>cost savings</text>

      <text x="1210" y="647" fontSize="88" fontWeight="800" letterSpacing="-3" fill={blue}>
        {`~${meanReduction}%`}
      </text>
      <text x="1210" y="705" fontSize="32" fill={ink}>lower mean latency</text>
      <text x="1210" y="749" fontSize="32" fill={muted}>vs direct Vega</text>

      <text x="70" y="807" fontSize="32" fill={muted}>
        {`${evidence.quality_items} historical public JevBench items · not official v1.6.1 · warm concurrency 1`}
      </text>
      <text x="70" y="853" fontSize="32" fill={muted}>Cost vs Vega-only: equal cost per Vega call, Kai cost ≈ 0. p95 is higher.</text>
    </svg>
  )
}
