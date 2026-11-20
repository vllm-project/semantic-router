import { describe, expect, it } from 'vitest'

import { requestStatusText } from './insightsDisplayFormat'

describe('requestStatusText', () => {
  it('shows the HTTP status of a request that ended the way it says', () => {
    expect(requestStatusText(200, { successful: true, state: 'completed' })).toEqual({
      text: '200',
      code: true,
    })
    expect(requestStatusText(502, { successful: false, state: 'failed' })).toEqual({
      text: '502',
      code: true,
    })
  })

  it('shows the lifecycle of a request that failed after a success status', () => {
    expect(requestStatusText(200, { successful: false, state: 'failed' })).toEqual({
      text: 'Failed',
      code: false,
      hiddenStatus: 200,
    })
    expect(requestStatusText(200, { successful: false, state: 'aborted' })).toMatchObject({
      text: 'Aborted',
      hiddenStatus: 200,
    })
  })

  it('shows the lifecycle of a request without a status', () => {
    expect(requestStatusText(undefined, { successful: false, state: 'client_closed' })).toEqual({
      text: 'Client closed',
      code: false,
      hiddenStatus: undefined,
    })
  })
})
