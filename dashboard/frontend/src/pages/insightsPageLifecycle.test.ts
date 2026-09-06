import { describe, expect, it } from 'vitest'

import { getInsightsLifecyclePresentation } from './insightsPageSupport'
import type { InsightsRecord } from './insightsPageTypes'

const record = (lifecycle_state: InsightsRecord['lifecycle_state'], response_status = 200) =>
  ({ lifecycle_state, response_status }) as InsightsRecord

describe('Insights lifecycle presentation', () => {
  it('shows completed 2xx records as successful', () => {
    expect(getInsightsLifecyclePresentation(record('completed'))).toMatchObject({
      successful: true,
      errored: false,
      pending: false,
    })
  })

  it('shows aborted streams as terminal errors, not pending work', () => {
    expect(getInsightsLifecyclePresentation(record('aborted'))).toMatchObject({
      successful: false,
      errored: true,
      pending: false,
    })
  })

  it('reserves pending presentation for in-progress records', () => {
    expect(getInsightsLifecyclePresentation(record('in_progress', 0))).toMatchObject({
      successful: false,
      errored: false,
      pending: true,
    })
  })
})
