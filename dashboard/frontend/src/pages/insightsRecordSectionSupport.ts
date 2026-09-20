export type RecordSectionSize = 'compact' | 'half' | 'feature' | 'wide'

const SECTION_PRESENTATION: Record<
  string,
  {
    size: RecordSectionSize
    collapsible?: boolean
    defaultExpanded?: boolean
    structured?: boolean
    description?: string
    metricColumns?: number
    noteFields?: string[]
    secondaryFields?: string[]
    secondaryTitle?: string
  }
> = {
  Lifecycle: { size: 'compact' },
  'Decision Information': { size: 'compact' },
  'Model Selection': { size: 'compact' },
  'Selection Stages': {
    size: 'wide',
    collapsible: true,
    defaultExpanded: true,
    structured: true,
    description: 'Priority filters and measured candidate evidence',
  },
  'Session Routing': { size: 'wide' },
  'Protection Candidate Scores': { size: 'wide', collapsible: true, defaultExpanded: true },
  'Observed Candidate Scores': { size: 'wide', collapsible: true, defaultExpanded: true },
  'Adaptation Candidate Scores': { size: 'wide', collapsible: true, defaultExpanded: true },
  'Request Capacity': { size: 'wide', collapsible: true, defaultExpanded: false },
  'Usage & Cost': {
    size: 'wide',
    description: 'Recorded token usage and configured-rate estimates',
    metricColumns: 4,
    noteFields: ['Cost basis'],
    secondaryFields: ['Baseline basis', 'Current pricing'],
    secondaryTitle: 'How these estimates are calculated',
  },
  Signals: { size: 'half' },
  'Plugin Status': { size: 'half', metricColumns: 3 },
  'Routing Metadata': {
    size: 'wide',
    collapsible: true,
    defaultExpanded: false,
    structured: true,
    description: 'Captured projection outputs and signal measurements',
  },
  'Projection Trace': {
    size: 'wide',
    collapsible: true,
    defaultExpanded: true,
    structured: true,
    description: 'Signal groups → weighted scores → routing outputs',
  },
  'Request / Response': { size: 'wide', collapsible: true, defaultExpanded: false },
}

export function getInsightsRecordSectionPresentation(title?: string) {
  return SECTION_PRESENTATION[title || ''] || { size: 'compact' as const }
}
