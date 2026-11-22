import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { ReviewActivatePanel } from './SetupWizardReviewPanel'

const renderPanel = (overrides: {
  isReadonly?: boolean
  activationBlockedReason?: string | null
  activationNotice?: string | null
}) =>
  renderToStaticMarkup(
    createElement(ReviewActivatePanel, {
      routeLabel: 'From scratch',
      decisionModelLabel: 'Vela 3.0 fast',
      modelNames: ['qwen/qwen3.5-rocm'],
      providers: ['vllm'],
      listenerPort: 8899,
      validationState: 'valid',
      validationError: null,
      activationState: 'idle',
      activationError: null,
      validatedCounts: { models: 1, decisions: 1, signals: 0, canActivate: true },
      modelsCount: 1,
      generatedDecisions: 1,
      generatedSignals: 0,
      previewYaml: 'version: v0.3',
      readonlyLoading: false,
      isReadonly: overrides.isReadonly ?? false,
      activationBlockedReason: overrides.activationBlockedReason ?? null,
      activationNotice: overrides.activationNotice ?? null,
      onValidateAgain: () => undefined,
      onEdit: () => undefined,
    }),
  )

describe('setup review panel activation states', () => {
  it('explains why the setup capability blocks activation', () => {
    const markup = renderPanel({
      activationBlockedReason: 'Your role lacks the config.deploy permission.',
    })
    expect(markup).toContain('Your role lacks the config.deploy permission.')
  })

  it('shows one read-only message, not the capability remediation too', () => {
    const markup = renderPanel({
      isReadonly: true,
      activationBlockedReason:
        'The Dashboard runs in read-only mode; setup activation is disabled.',
    })
    expect(markup).toContain('running in read-only mode')
    expect(markup).not.toContain('setup activation is disabled.')
  })

  it('shows a degraded capability caveat only while activation is allowed', () => {
    const notice = 'Activation will be saved; run `vllm-sr serve` to start the Router.'
    expect(renderPanel({ activationNotice: notice })).toContain('run `vllm-sr serve`')
    expect(
      renderPanel({
        activationNotice: notice,
        activationBlockedReason: 'Checking setup activation access…',
      }),
    ).not.toContain('run `vllm-sr serve`')
  })

  it('disables Activate while the setup capability blocks it', () => {
    const source = readFileSync(new URL('./SetupWizardPage.tsx', import.meta.url), 'utf8')
    expect(source).toContain('activationBlockedReason !== null')
    expect(source).toMatch(/const setupCapability = useCapability\(['"]setup['"]\)/)
  })
})
