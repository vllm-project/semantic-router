import { describe, expect, it } from 'vitest'

import type { ServiceStatus } from '../utils/routerRuntime'
import { clampPage, filterServices } from './statusPageSupport'

const services: ServiceStatus[] = [
  { name: 'router', component: 'semantic-router', status: 'ready', healthy: true },
  { name: 'envoy', status: 'degraded', message: 'upstream unavailable', healthy: false },
]

describe('status page support', () => {
  it('filters services by health and searchable metadata', () => {
    expect(filterServices(services, '', 'unhealthy')).toEqual([services[1]])
    expect(filterServices(services, 'semantic', 'all')).toEqual([services[0]])
    expect(filterServices(services, 'upstream', 'all')).toEqual([services[1]])
  })

  it('clamps pagination after result counts change', () => {
    expect(clampPage(4, 13, 6)).toBe(3)
    expect(clampPage(0, 0, 6)).toBe(1)
  })
})
