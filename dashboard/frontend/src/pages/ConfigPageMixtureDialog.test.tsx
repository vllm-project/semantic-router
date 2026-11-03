import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ConfigPageMixtureDialog from './ConfigPageMixtureDialog'
import type { BuiltInRecipeCatalog } from './configPageBuiltInRecipes'
import type { ConfigData, EntrypointConfig } from './configPageSupport'

const catalog: BuiltInRecipeCatalog = {
  catalog_version: 'latest',
  bundles: [
    {
      name: 'mom-v1',
      sha256: 'sha256:aaaa',
      recipes: [
        {
          name: 'vault',
          description: 'Keep private traffic inside the deployment boundary.',
          decisions: [
            {
              name: 'guard',
              description: 'Contain prompt attacks.',
              algorithm: null,
              minimum_candidates: 0,
            },
            {
              name: 'private',
              description: 'Serve private requests.',
              algorithm: 'multi_factor',
              minimum_candidates: 1,
            },
          ],
        },
      ],
    },
  ],
}
const entrypoint: EntrypointConfig = { model_names: ['my/blend'], recipe: 'blend' }
const config: ConfigData = {
  providers: { models: [{ name: 'model-a' }] },
  entrypoints: [entrypoint],
  recipes: [
    {
      name: 'blend',
      routing: {
        decisions: [
          {
            name: 'answer',
            description: 'Answer every request.',
            priority: 1,
            rules: { operator: 'AND', conditions: [] },
            modelRefs: [{ model: 'model-a', use_reasoning: false }],
          },
        ],
      },
    },
  ],
}

const render = (props: Partial<Parameters<typeof ConfigPageMixtureDialog>[0]>) =>
  renderToStaticMarkup(
    <ConfigPageMixtureDialog
      config={config}
      models={[{ name: 'model-a', endpoints: [] }]}
      builtInRecipes={catalog}
      onClose={vi.fn()}
      onSave={vi.fn()}
      {...props}
    />,
  )

describe('ConfigPageMixtureDialog', () => {
  it('groups built-in recipes and asks for models only where a decision calls one', () => {
    const markup = render({ initialRecipe: { bundle: 'mom-v1', name: 'vault' } })

    expect(markup).toContain('<optgroup label="Configured recipes">')
    expect(markup).toContain('<optgroup label="Built-in · mom-v1">')
    expect(markup).toMatch(/<option value="[^"]*mom-v1[^"]*vault[^"]*" selected="">vault<\/option>/)
    expect(markup).toContain('Keep private traffic inside the deployment boundary.')
    expect(markup).toContain('1 of 2 ready')
    expect(markup).toContain('No model needed')
    expect(markup).toContain('This decision responds without calling a model.')
  })

  it('edits an entrypoint with config recipes only', () => {
    const markup = render({ entrypoint, entrypointIndex: 0 })

    expect(markup).not.toContain('Built-in')
    expect(markup).toMatch(/<option value="[^"]*blend[^"]*" selected="">blend<\/option>/)
    expect(markup).toContain('1 selected · minimum 1')
  })
})
