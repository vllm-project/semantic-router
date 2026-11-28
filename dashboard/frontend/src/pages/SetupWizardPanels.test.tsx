import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '../ui'
import { ModelStepPanel, RoutingStarterPanel, SetupWizardStepRail } from './SetupWizardPanels'
import { createModelDraft, createSetupConfigCounts } from './setupWizardSupport'

describe('setup wizard panels', () => {
  it('renders a current-aware keyboard stepper with controlled regions', () => {
    const markup = renderToStaticMarkup(
      createElement(SetupWizardStepRail, {
        currentStep: 1,
        finished: false,
        summaries: ['1 model connected', 'From scratch', 'Vela 3.0 fast', 'Writes config.yaml'],
        onGoToStep: () => true,
      }),
    )

    expect(markup).toContain('<nav')
    expect(markup).toContain('aria-label="Setup progress"')
    expect(markup).toContain('aria-current="step"')
    expect(markup).toContain('aria-controls="setup-step-1-panel"')
    expect(markup).toContain('tabindex="0"')
    expect(markup).toContain('Step 1: Connect models, complete')
    expect(markup).toContain('Step 4: Review &amp; activate')
  })

  it('bounds a large model collection and exposes search, count, and paging', () => {
    const models = Array.from({ length: 6 }, (_, index) => ({
      ...createModelDraft(index + 1),
      id: `model-${index + 1}`,
      name: `model-${index + 1}`,
      baseUrl: `vllm:${8000 + index}`,
    }))
    const markup = renderToStaticMarkup(
      createElement(
        TooltipProvider,
        null,
        createElement(ModelStepPanel, {
          models,
          defaultModelId: 'model-1',
          shouldShowStepOneIssues: false,
          stepOneErrors: [],
          stepOneAttempted: false,
          draftBuildError: null,
          removedModel: null,
          onAddModel: vi.fn(),
          onUpdateModel: vi.fn(),
          onRemoveModel: vi.fn(),
          onUndoRemoveModel: vi.fn(),
          onSelectDefaultModel: vi.fn(),
        }),
      ),
    )

    expect(markup).toContain('type="search"')
    expect(markup).toContain('6 models')
    expect(markup).toContain('Page 1 of 2')
    expect(markup).toContain('Next models')
    expect(markup.match(/<article/g)).toHaveLength(4)
    expect(markup).toContain('Remove model model-1')
  })

  it('keeps provider secrets out of the browser login password manager', () => {
    const model = { ...createModelDraft(1), id: 'model-1', endpointName: 'primary' }
    const markup = renderToStaticMarkup(
      createElement(
        TooltipProvider,
        null,
        createElement(ModelStepPanel, {
          models: [model],
          defaultModelId: 'model-1',
          shouldShowStepOneIssues: false,
          stepOneErrors: [],
          stepOneAttempted: false,
          draftBuildError: null,
          removedModel: null,
          onAddModel: vi.fn(),
          onUpdateModel: vi.fn(),
          onRemoveModel: vi.fn(),
          onUndoRemoveModel: vi.fn(),
          onSelectDefaultModel: vi.fn(),
        }),
      ),
    )
    const inputTag = (id: string) => markup.match(new RegExp(`<input[^>]*id="${id}"[^>]*>`))?.[0]

    const key = inputTag('model-1-key')
    expect(key).toContain('type="password"')
    expect(key).toMatch(/autocomplete="new-password"/i)
    expect(key).toContain('name="model-1-provider-access-key"')
    for (const id of ['model-1-name', 'model-1-base-url', 'model-1-endpoint']) {
      expect(inputTag(id)).toMatch(/autocomplete="off"/i)
    }
  })

  it('makes remote and preset async failures recoverable', () => {
    const markup = renderToStaticMarkup(
      createElement(RoutingStarterPanel, {
        routingMode: 'remote',
        remoteConfigUrl: 'https://example.com/config.yaml',
        remoteImportState: 'error',
        remoteImportError: 'Network unavailable',
        importedConfig: null,
        counts: createSetupConfigCounts(),
        presets: [],
        presetCatalogState: 'error',
        presetCatalogError: 'Preset service unavailable',
        selectedPresetId: null,
        presetRequestState: 'idle',
        presetDelta: null,
        presetImportedConfig: null,
        presetError: null,
        onSelectRoutingMode: vi.fn(),
        onChangeRemoteConfigUrl: vi.fn(),
        onImportRemoteConfig: vi.fn(),
        onSelectPreset: vi.fn(),
        onImportPresetConfig: vi.fn(),
        onRetryPresets: vi.fn(),
      }),
    )

    expect(markup).toContain('Retry presets')
    expect(markup).toContain('Retry import')
    expect(markup).toContain('aria-invalid="true"')
    expect(markup).toContain('Preset service unavailable')
    expect(markup).toContain('Network unavailable')
  })

  it('uses shared confirmation and has no JSX inline-style escape hatches', () => {
    const source = readFileSync(new URL('./SetupWizardPanels.tsx', import.meta.url), 'utf8')

    expect(source).toContain('ConfirmDialog')
    expect(source).toContain('Remove this model?')
    expect(source).toContain('onUndoRemoveModel')
    expect(source).not.toContain('style={{')
  })
})
