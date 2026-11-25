import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { MixtureEditorBody } from './ConfigPageMixtureDialog'
import { useMixtureEditor, type ConfigPageMixtureDialogProps } from './useMixtureEditor'
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

function Harness(props: ConfigPageMixtureDialogProps) {
  const editor = useMixtureEditor(props)
  return <MixtureEditorBody editor={editor} />
}

const render = (props: Partial<ConfigPageMixtureDialogProps>) =>
  renderToStaticMarkup(
    <Harness
      config={config}
      models={[{ name: 'model-a', endpoints: [] }]}
      builtInRecipes={catalog}
      onClose={vi.fn()}
      onSave={vi.fn()}
      {...props}
    />,
  )

describe('public model editor', () => {
  it('groups built-in recipes and asks for models only where a decision calls one', () => {
    const markup = render({ initialRecipe: { bundle: 'mom-v1', name: 'vault' } })

    expect(markup).toContain('<optgroup label="Configured recipes">')
    expect(markup).toContain('<optgroup label="Built-in · mom-v1">')
    expect(markup).toMatch(/<option value="[^"]*mom-v1[^"]*vault[^"]*" selected="">vault<\/option>/)
    expect(markup).toContain('Keep private traffic inside the deployment boundary.')
    expect(markup).toContain('1 of 2 ready')
    expect(markup).toContain('No model needed')
    expect(markup).toContain('This decision responds without calling a model.')
    expect(markup).not.toContain('Reviews go to')
  })

  it('edits an entrypoint with config recipes only', () => {
    const markup = render({ entrypoint, entrypointIndex: 0 })

    expect(markup).not.toContain('Built-in')
    expect(markup).toMatch(/<option value="[^"]*blend[^"]*" selected="">blend<\/option>/)
    expect(markup).toContain('1 selected · minimum 1')
    expect(markup).toContain('value="my/blend"')
  })

  it('reports a name that another public model already claims', () => {
    const markup = render({
      entrypoint: { model_names: ['my/blend'], recipe: 'blend' },
      config: {
        ...config,
        entrypoints: [entrypoint, { model_names: ['taken'], recipe: 'blend' }],
      },
      entrypointIndex: undefined,
    })
    expect(markup).toContain('Entrypoint model &quot;my/blend&quot; is already mapped.')
  })
})

it('offers other entrypoints as reviewers and retains the chosen review model', () => {
  const edited = { ...entrypoint, review_model: 'review' }
  const html = render({
    entrypoint: edited,
    entrypointIndex: 0,
    config: {
      ...config,
      entrypoints: [
        edited,
        { model_names: ['review'], recipe: 'blend' },
        { model_names: ['chained'], recipe: 'blend', review_model: 'review' },
      ],
    },
  })
  expect(html).toContain('Reviews go to')
  expect(html).toContain('Session model')
  expect(html).toContain('<option value="review" selected="">review</option>')
  expect(html).not.toContain('<option value="my/blend"')
  expect(html).not.toContain('<option value="chained"')
})

it('offers other entrypoints for memory calls and retains the chosen memory model', () => {
  const edited = { ...entrypoint, memory_model: 'local' }
  const html = render({
    entrypoint: edited,
    entrypointIndex: 0,
    config: {
      ...config,
      entrypoints: [
        edited,
        { model_names: ['local'], recipe: 'blend' },
        { model_names: ['remembers'], recipe: 'blend', memory_model: 'local' },
      ],
    },
  })
  expect(html).toContain('Memories go to')
  expect(html).toContain('<option value="local" selected="">local</option>')
  // the review select offers the same target, not chosen there
  expect(html).toContain('<option value="local">local</option>')
  expect(html).not.toContain('<option value="remembers"')
})
