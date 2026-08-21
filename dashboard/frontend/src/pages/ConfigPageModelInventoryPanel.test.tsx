import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ConfigPageModelInventoryPanel from './ConfigPageModelInventoryPanel'
import type { NormalizedModel } from './configPageSupport'

const model: NormalizedModel = {
  name: 'remote/example-model',
  provider_model_id: 'example-model',
  endpoints: [
    {
      name: 'primary',
      endpoint: 'https://models.example.test/v1',
      protocol: 'http',
      weight: 1,
    },
  ],
}

function renderInventory(canVerifyModels: boolean) {
  const noop = vi.fn()

  return renderToStaticMarkup(
    <ConfigPageModelInventoryPanel
      models={[model]}
      filteredModels={[model]}
      defaultModel=""
      modelReferenceCounts={new Map()}
      modelsSearch=""
      onModelsSearchChange={noop}
      reasoningFamilyFilter="all"
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
      expandedModels={new Set()}
      onToggleExpand={noop}
      renderExpandedRow={() => null}
      getDeleteBlocker={() => null}
      liveVerificationStates={new Map()}
      onVerifyModel={noop}
      canVerifyModels={canVerifyModels}
    />,
  )
}

describe('ConfigPageModelInventoryPanel live verification visibility', () => {
  it('omits the Live column for read-only dashboard users', () => {
    expect(renderInventory(false)).not.toContain('>Live<')
  })

  it('renders the Live column for operators and administrators', () => {
    expect(renderInventory(true)).toContain('>Live<')
  })
})
