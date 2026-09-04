import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { CopyableValuePresentation } from './EvaluationCampaignDecisionTechnicalFields'
import { copyTextToClipboard } from './evaluationTechnicalFields'

describe('EvaluationCampaignDecisionTechnicalFields clipboard feedback', () => {
  it('turns a rejected clipboard write into an announced retry action', async () => {
    const writeText = vi.fn().mockRejectedValue(new Error('Clipboard permission denied.'))

    const copyState = await copyTextToClipboard('run-123', { writeText })
    const markup = renderToStaticMarkup(
      createElement(CopyableValuePresentation, {
        label: 'run ID',
        value: 'run-123',
        copyState,
        onCopy: () => undefined,
      }),
    )

    expect(writeText).toHaveBeenCalledWith('run-123')
    expect(copyState).toBe('failed')
    expect(markup).toContain('aria-label="Retry copy run ID"')
    expect(markup).toContain('aria-live="polite"')
    expect(markup).toContain('run ID could not be copied.')
  })
})
