import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { HubHero } from './ModelHubComponents'

describe('Model Hub hero', () => {
  it('presents every provider without exposing internal mapping classes', () => {
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <HubHero
          stats={{
            models: 101,
            physicalModels: 90,
            virtualModels: 11,
            mappedProviders: 3,
            providerContracts: 17,
            creators: 24,
            evaluatedModels: 80,
            evaluations: 900,
          }}
        />
      </MemoryRouter>,
    )

    expect(markup).toContain('<dd>17</dd><dt>providers</dt>')
    expect(markup).not.toContain('mapped providers')
  })
})
