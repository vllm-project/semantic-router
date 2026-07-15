import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import {
  ModelStepPanel,
  RoutingStarterPanel,
  SetupWizardStepper,
} from './SetupWizardPanels'
import { createModelDraft, createSetupConfigCounts } from './setupWizardSupport'

describe('setup wizard panels', () => {
  it('renders a current-aware keyboard stepper with controlled regions', () => {
    const markup = renderToStaticMarkup(
      createElement(SetupWizardStepper, {
        currentStep: 1,
        onGoToStep: () => true,
      }),
    )

    expect(markup).toContain('<nav')
    expect(markup).toContain('aria-label="Setup progress"')
    expect(markup).toContain('aria-current="step"')
    expect(markup).toContain('aria-controls="setup-step-1-panel"')
    expect(markup).toContain('tabindex="0"')
    expect(markup).toContain('Step 1: Connect model, complete')
  })

  it('bounds a large model collection and exposes search, count, and paging', () => {
    const models = Array.from({ length: 6 }, (_, index) => ({
      ...createModelDraft(index + 1),
      id: `model-${index + 1}`,
      name: `model-${index + 1}`,
      baseUrl: `vllm:${8000 + index}`,
    }))
    const markup = renderToStaticMarkup(
      createElement(ModelStepPanel, {
        currentRouteLabel: 'From scratch',
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
    )

    expect(markup).toContain('type="search"')
    expect(markup).toContain('6 models')
    expect(markup).toContain('Page 1 of 2')
    expect(markup).toContain('Next models')
    expect(markup.match(/<article/g)).toHaveLength(4)
    expect(markup).toContain('Remove model model-1')
  })

  it('makes remote and preset async failures recoverable', () => {
    const markup = renderToStaticMarkup(
      createElement(RoutingStarterPanel, {
        currentRouteLabel: 'From remote',
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
