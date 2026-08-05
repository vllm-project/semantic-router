import type { KnowledgeBaseManagerView } from './configPageKnowledgeBaseManagerSupport'

interface KnowledgeBaseCounts {
  total: number
  builtin: number
  custom: number
}

interface GroupOverview {
  total: number
  referenced: number
  metricBacked: number
}

interface LabelOverview {
  total: number
  referenced: number
  overrides: number
}

export interface TaxonomySummaryCard {
  label: string
  value: number
  hint: string
}

interface BuildTaxonomySummaryCardsOptions {
  activeView: KnowledgeBaseManagerView
  counts: KnowledgeBaseCounts
  groupOverview: GroupOverview
  labelOverview: LabelOverview
}

export function buildTaxonomySummaryCards({
  activeView,
  counts,
  groupOverview,
  labelOverview,
}: BuildTaxonomySummaryCardsOptions): TaxonomySummaryCard[] {
  if (activeView === 'groups') {
    return [
      {
        label: 'Groups',
        value: groupOverview.total,
        hint: 'Groups defined in the active knowledge base.',
      },
      {
        label: 'Signal-backed',
        value: groupOverview.referenced,
        hint: 'Groups currently referenced by routing signals.',
      },
      {
        label: 'Metric-backed',
        value: groupOverview.metricBacked,
        hint: 'Groups used by KB metrics.',
      },
    ]
  }
  if (activeView === 'labels') {
    return [
      {
        label: 'Labels',
        value: labelOverview.total,
        hint: 'Labels available in the active knowledge base.',
      },
      {
        label: 'Signal-backed',
        value: labelOverview.referenced,
        hint: 'Labels currently referenced by routing signals.',
      },
      {
        label: 'Threshold Overrides',
        value: labelOverview.overrides,
        hint: 'Labels overriding the base threshold.',
      },
    ]
  }
  return [
    {
      label: 'Total Knowledge Bases',
      value: counts.total,
      hint: 'All KB packages currently discoverable by the router.',
    },
    {
      label: 'Built-In',
      value: counts.builtin,
      hint: 'Router-shipped bases managed through the same control plane.',
    },
    {
      label: 'Custom',
      value: counts.custom,
      hint: 'User-managed bases persisted in the shared runtime KB store.',
    },
  ]
}
