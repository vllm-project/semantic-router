import { describe, expect, it } from 'vitest'

import styles from './InsightsPage.module.css'
import {
  getInsightsLifecyclePresentation,
  getInsightsLifecycleStatusClass,
} from './insightsPageSupport'
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

  it('maps successful records onto the shared success status class', () => {
    expect(
      getInsightsLifecycleStatusClass(getInsightsLifecyclePresentation(record('completed'))),
    ).toBe(styles.statusSuccess)
  })

  it('maps failed records onto the shared error status class', () => {
    expect(
      getInsightsLifecycleStatusClass(getInsightsLifecyclePresentation(record('failed'))),
    ).toBe(styles.statusError)
  })

  it('maps in-progress records onto the shared pending status class', () => {
    expect(
      getInsightsLifecycleStatusClass(getInsightsLifecyclePresentation(record('in_progress', 0))),
    ).toBe(styles.statusPending)
  })
})
