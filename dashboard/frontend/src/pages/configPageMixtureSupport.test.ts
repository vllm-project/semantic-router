import { describe, expect, it } from 'vitest'

import {
  applyRecipeAssignments,
  assignmentState,
  assignDecisionModels,
  minimumCandidatesForDecision,
} from './configPageMixtureSupport'
import type { ConfigData, DecisionConfig } from './configPageSupport'

const decisions: DecisionConfig[] = [
  {
    name: 'Simple',
    description: 'Fast path',
    priority: 10,
    rules: { operator: 'AND', conditions: [] },
    modelRefs: [{ model: 'local/fast', use_reasoning: false, weight: 2 }],
  },
  {
    name: 'Complex',
    description: 'Frontier path',
    priority: 20,
    rules: { operator: 'AND', conditions: [] },
    modelRefs: [{ model: 'remote/frontier', use_reasoning: true }],
  },
]

describe('Mixture model assignments', () => {
  it('reads assignments without changing their decision order', () => {
    expect(assignmentState(decisions)).toEqual({
      Simple: ['local/fast'],
      Complex: ['remote/frontier'],
    })
  })

  it('uses the decision candidate contract and defaults legacy recipes to one', () => {
    expect(minimumCandidatesForDecision(decisions[0])).toBe(1)
    expect(
      minimumCandidatesForDecision({
        ...decisions[1],
        algorithm: { type: 'fusion', minimum_candidates: 3 },
      }),
    ).toBe(3)
  })

  it('preserves existing model-ref policy and initializes only new references', () => {
    const updated = assignDecisionModels(decisions, {
      Simple: ['local/fast', 'local/balanced'],
      Complex: ['remote/frontier'],
    })

    expect(updated[0].modelRefs).toEqual([
      { model: 'local/fast', use_reasoning: false, weight: 2 },
      { model: 'local/balanced', use_reasoning: false },
    ])
    expect(decisions[0].modelRefs).toEqual([
      { model: 'local/fast', use_reasoning: false, weight: 2 },
    ])
  })

  it('materializes a missing dynamic workflow planner from the first assignment', () => {
    const workflow: DecisionConfig = {
      name: 'Orchestrate',
      description: 'Dynamic workflow',
      priority: 30,
      rules: { operator: 'AND', conditions: [] },
      modelRefs: [],
      algorithm: {
        type: 'workflows',
        workflows: {
          mode: 'dynamic',
          planner: { max_completion_tokens: 1024 },
        },
      },
    }

    const [updated] = assignDecisionModels([workflow], {
      Orchestrate: ['local/coordinator', 'local/worker'],
    })

    expect(updated.algorithm).toEqual({
      type: 'workflows',
      workflows: {
        mode: 'dynamic',
        planner: {
          model: 'local/coordinator',
          max_completion_tokens: 1024,
        },
      },
    })
    expect(workflow.algorithm).not.toHaveProperty('workflows.planner.model')
  })

  it('preserves an explicitly configured dynamic workflow planner', () => {
    const workflow: DecisionConfig = {
      name: 'Orchestrate',
      description: 'Dynamic workflow',
      priority: 30,
      rules: { operator: 'AND', conditions: [] },
      modelRefs: [],
      algorithm: {
        type: 'workflows',
        workflows: {
          mode: 'dynamic',
          planner: { model: 'external/coordinator' },
        },
      },
    }

    const [updated] = assignDecisionModels([workflow], {
      Orchestrate: ['local/worker'],
    })

    expect(updated.algorithm).toHaveProperty('workflows.planner.model', 'external/coordinator')
  })

  it('updates only the selected named recipe', () => {
    const config: ConfigData = {
      recipes: [
        { name: 'blend', routing: { decisions } },
        { name: 'private', routing: { decisions: structuredClone(decisions) } },
      ],
    }
    const updated = applyRecipeAssignments(config, 'blend', {
      Simple: ['local/balanced'],
      Complex: ['remote/frontier'],
    })

    expect(updated.recipes?.[0].routing.decisions?.[0].modelRefs?.[0].model).toBe('local/balanced')
    expect(updated.recipes?.[1]).toEqual(config.recipes?.[1])
    expect(config.recipes?.[0].routing.decisions?.[0].modelRefs?.[0].model).toBe('local/fast')
  })

  it('writes the synthetic default recipe back to top-level routing', () => {
    const config: ConfigData = { decisions }
    const updated = applyRecipeAssignments(config, 'default', {
      Simple: ['local/balanced'],
      Complex: ['remote/frontier'],
    })

    expect(updated.routing?.decisions?.[0].modelRefs?.[0].model).toBe('local/balanced')
    expect(updated.decisions).toBeUndefined()
  })
})
