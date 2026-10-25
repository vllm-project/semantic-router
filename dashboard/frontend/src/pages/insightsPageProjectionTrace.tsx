import type { ViewField } from '../components/ViewPanel'
import InsightsProjectionTrace from './InsightsProjectionTrace'
import type { InsightsRecord } from './insightsPageTypes'

export function buildProjectionTraceFields(record: InsightsRecord): ViewField[] {
  return record.projection_trace
    ? [
        {
          label: 'Recorded projection stages',
          fullWidth: true,
          value: <InsightsProjectionTrace trace={record.projection_trace} recordID={record.id} />,
        },
      ]
    : []
}
