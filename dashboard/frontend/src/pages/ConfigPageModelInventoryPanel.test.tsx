import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import ConfigPageModelInventoryPanel from './ConfigPageModelInventoryPanel'
import type { NormalizedModel } from './configPageSupport'

const noop = () => {}

function render(models: NormalizedModel[], catalogPrices = {}) {
  return renderToStaticMarkup(
    <ConfigPageModelInventoryPanel
      models={models}
      filteredModels={models}
      defaultModel=""
      modelReferenceCounts={new Map()}
      modelsSearch=""
      onModelsSearchChange={noop}
      reasoningFamilyFilter=""
      onReasoningFamilyFilterChange={noop}
      reasoningFamilyOptions={[]}
      endpointFilter="all"
      onEndpointFilterChange={noop}
      roleFilter="all"
      onRoleFilterChange={noop}
      filtersActive={false}
      onClearFilters={noop}
      isReadonly
      selectedModelKeys={new Set()}
      onSelectedModelKeysChange={noop}
      onClearSelection={noop}
      onDeleteSelected={noop}
      operationError={null}
      onDismissOperationError={noop}
      onAddModel={noop}
      onViewModel={noop}
      onEditModel={noop}
      onDeleteModel={noop}
      expandedModels={new Set()}
      onToggleExpand={noop}
      renderExpandedRow={() => null}
      getDeleteBlocker={() => null}
      liveVerificationStates={new Map()}
      onVerifyModel={noop}
      canVerifyModels={false}
      catalogPrices={catalogPrices}
    />,
  )
}

// The text of each row's Pricing cell, the column 132px wide.
function pricingCells(html: string): string[] {
  return [
    ...html.matchAll(/<td[^>]*width:132px;min-width:124px;text-align:left">(.*?)<\/td>/g),
  ].map((match) => match[1].replace(/<[^>]+>/g, ' ').trim())
}

describe('model inventory pricing', () => {
  it('shows the catalog list price of a model that sets no pricing and says whose it is', () => {
    const html = render(
      [
        { name: 'listed', endpoints: [] },
        { name: 'own', endpoints: [], pricing: { prompt_per_1m: 1 } },
      ],
      {
        listed: { currency: 'USD', prompt_per_1m: 2, completion_per_1m: 10 },
        own: { currency: 'USD', prompt_per_1m: 9 },
      },
    )
    expect(pricingCells(html)).toEqual(['$2.00 / 1M catalog list price', '$1.00 / 1M'])
    expect(html).toContain('per 1M input tokens, the catalog list price')
  })

  it('keeps N/A for a model with no price at all', () => {
    expect(pricingCells(render([{ name: 'local', endpoints: [] }]))).toEqual(['N/A'])
  })
})
