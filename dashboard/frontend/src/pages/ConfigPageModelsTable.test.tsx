import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import catalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { TooltipProvider } from '../ui'

import ConfigPageModelsTable from './ConfigPageModelsTable'
import type { NormalizedModel } from './configPageSupport'

const noop = () => {}

function render(models: NormalizedModel[], catalogPrices = {}) {
  return renderToStaticMarkup(
    <TooltipProvider>
      <ConfigPageModelsTable
        models={models}
        totalModels={models.length}
        defaultModel=""
        deployments={new Map()}
        referenceCounts={new Map()}
        kinds={new Map()}
        catalog={catalog as unknown as BuiltInModelCatalog}
        health={{
          phase: 'unavailable',
          byModel: new Map(),
          intervalSeconds: 30,
          checking: false,
          checkingModels: new Set(),
        }}
        traffic={{
          phase: 'unavailable',
          requests: 0,
          shares: new Map(),
          complete: true,
          spend: null,
          savedPercent: null,
        }}
        catalogPrices={catalogPrices}
        isReadonly
        activeName={null}
        resetKey=""
        filtersActive={false}
        selectedKeys={new Set()}
        onSelectedKeysChange={noop}
        getDeleteBlocker={() => null}
        canBeDefault={() => false}
        onOpen={noop}
        onShowUsage={noop}
        onEdit={noop}
        onDelete={noop}
        onMakeDefault={noop}
        onCheckHealth={noop}
        onConnect={noop}
        onClearFilters={noop}
      />
    </TooltipProvider>,
  )
}

describe('model prices', () => {
  it('shows the catalog list price of a model that sets no pricing and says whose it is', () => {
    const html = render(
      [
        { name: 'listed', endpoints: [] },
        { name: 'own', endpoints: [], pricing: { prompt_per_1m: 1, completion_per_1m: 4 } },
      ],
      {
        listed: { currency: 'USD', prompt_per_1m: 2, completion_per_1m: 10 },
        own: { currency: 'USD', prompt_per_1m: 9, completion_per_1m: 9 },
      },
    )
    // Only the model without pricing takes the catalog's price.
    expect(html.match(/catalog list price<\/span>/g)).toHaveLength(1)
    expect(html).toContain('per one million tokens, the catalog list price')
    expect(html).toContain('$2.00')
    expect(html).toContain('$10.00')
    expect(html).toContain('$4.00')
    expect(html).not.toContain('$9.00')
  })

  it('shows no price for a model the catalog does not price either', () => {
    const html = render([{ name: 'local', endpoints: [] }])
    expect(html).not.toContain('catalog list price')
  })
})
