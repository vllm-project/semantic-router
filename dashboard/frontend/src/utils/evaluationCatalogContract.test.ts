import { describe, expect, it } from 'vitest'

import {
  catalogWithUnavailableMixture,
  evaluationCatalogFixture,
  mixture,
} from '../test/evaluationPlaneApiFixture'
import { buildEvaluationRoutingRecipePlan } from '../test/evaluationRoutingRecipeFixture'
import { decodeEvaluationCatalog } from './evaluationCatalogContract'

describe('evaluation catalog contract', () => {
  it('decodes catalog targets with labeled complexity signals', () => {
    const catalog = evaluationCatalogFixture()
    const labeledPlan = buildEvaluationRoutingRecipePlan(
      mixture,
      [{ id: 'complexity:balance_difficulty:easy', value_kind: 'numeric' }],
      [],
    )
    const target = {
      ...catalogWithUnavailableMixture.targets[0],
      id: mixture.id,
      name: mixture.entrypoint_model,
      healthy: true,
      track_ids: ['routing'] as const,
      mixture: { ...mixture, routing_recipe_plan: labeledPlan },
    }

    expect(decodeEvaluationCatalog({ ...catalog, targets: [target] }).targets).toHaveLength(1)
  })
})
