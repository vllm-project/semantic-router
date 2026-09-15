export type RecordSectionSize = 'compact' | 'feature' | 'wide'

const SECTION_PRESENTATION: Record<
  string,
  { size: RecordSectionSize; collapsible?: boolean; defaultExpanded?: boolean }
> = {
  Lifecycle: { size: 'compact' },
  'Decision Information': { size: 'compact' },
  'Model Selection': { size: 'compact' },
  'Session Routing': { size: 'wide' },
  'Protection Candidate Scores': { size: 'wide', collapsible: true, defaultExpanded: true },
  'Observed Candidate Scores': { size: 'wide', collapsible: true, defaultExpanded: true },
  'Adaptation Candidate Scores': { size: 'wide', collapsible: true, defaultExpanded: true },
  'Request Capacity': { size: 'wide', collapsible: true, defaultExpanded: false },
  'Usage & Cost': { size: 'compact' },
  Signals: { size: 'compact' },
  'Plugin Status': { size: 'compact' },
  'Routing Metadata': { size: 'wide', collapsible: true, defaultExpanded: false },
  'Projection Trace': { size: 'wide', collapsible: true, defaultExpanded: true },
  'Request / Response': { size: 'wide', collapsible: true, defaultExpanded: false },
}

export function getInsightsRecordSectionPresentation(title?: string) {
  return SECTION_PRESENTATION[title || ''] || { size: 'compact' as const }
}
