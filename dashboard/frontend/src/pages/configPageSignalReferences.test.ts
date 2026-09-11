import { describe, expect, it } from 'vitest'

import type { ConfigData, RecipeRoutingConfig } from './configPageSupport'
import {
  getSignalReferenceCount,
  getSignalReferenceCountInRoutingProfile,
} from './configPageSignalReferences'

describe('signal references', () => {
  it('counts decision, projection, and composer references in one routing scope', () => {
    const routing: RecipeRoutingConfig = {
      signals: {
        complexity: [
          {
            name: 'difficulty',
            threshold: 0.5,
            hard: { candidates: [] },
            easy: { candidates: [] },
            composer: {
              operator: 'AND',
              conditions: [{ type: 'event', name: 'release' }],
            },
          },
        ],
      },
      projections: {
        scores: [
          {
            name: 'event-score',
            method: 'weighted_sum',
            inputs: [{ type: 'event', name: 'release', weight: 1 }],
          },
        ],
      },
      decisions: [
        {
          name: 'event-route',
          description: '',
          priority: 1,
          rules: {
            operator: 'AND',
            conditions: [{ type: 'event', name: 'release' }],
          },
          modelRefs: [],
        },
      ],
    }

    expect(getSignalReferenceCountInRoutingProfile(routing, 'Event', 'release')).toBe(3)
  })

  it('keeps reference counts local to the selected recipe', () => {
    const config: ConfigData = {
      recipes: [
        { name: 'alpha', routing: { decisions: [] } },
        {
          name: 'beta',
          routing: {
            decisions: [
              {
                name: 'image-route',
                description: '',
                priority: 1,
                rules: {
                  operator: 'AND',
                  conditions: [{ type: 'input_modality', name: 'has-image' }],
                },
                modelRefs: [],
              },
            ],
          },
        },
      ],
    }

    expect(
      getSignalReferenceCountInRoutingProfile(
        config.recipes?.[0].routing,
        'Input Modality',
        'has-image',
      ),
    ).toBe(0)
    expect(
      getSignalReferenceCountInRoutingProfile(
        config.recipes?.[1].routing,
        'Input Modality',
        'has-image',
      ),
    ).toBe(1)
  })

  it('counts shared and recipe references across the complete configuration', () => {
    const config: ConfigData = {
      routing: {
        decisions: [
          {
            name: 'default-route',
            description: '',
            priority: 1,
            rules: {
              operator: 'AND',
              conditions: [{ type: 'metadata', name: 'tenant' }],
            },
            modelRefs: [],
          },
        ],
      },
      recipes: [
        {
          name: 'private',
          routing: {
            decisions: [
              {
                name: 'private-route',
                description: '',
                priority: 1,
                rules: {
                  operator: 'AND',
                  conditions: [{ type: 'metadata', name: 'tenant' }],
                },
                modelRefs: [],
              },
            ],
          },
        },
      ],
    }

    expect(getSignalReferenceCount(config, 'Metadata', 'tenant')).toBe(2)
  })
})
