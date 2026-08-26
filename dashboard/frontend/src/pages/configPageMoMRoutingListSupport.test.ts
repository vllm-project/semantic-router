import { describe, expect, it } from 'vitest'

import type { RoutingEntrypoint } from '../utils/routingManagementApi'
import { assignedModelCount } from './configPageMoMRoutingListSupport'

const entrypoint = (id: string, assignedModels: number): RoutingEntrypoint => ({
  id,
  name: id,
  status: 'active',
  revision: 1,
  entrypointRevision: 1,
  aliases: [id],
  recipeIds: [],
  ruleCount: 2,
  assignedModelCount: assignedModels,
  createdAt: '2026-08-23T00:00:00Z',
  updatedAt: '2026-08-23T00:00:00Z',
})

describe('Mixture-of-Models list counts', () => {
  it('uses the bounded list summary without loading topology', () => {
    expect(assignedModelCount(entrypoint('mom-summary', 4))).toBe(4)
  })
})
