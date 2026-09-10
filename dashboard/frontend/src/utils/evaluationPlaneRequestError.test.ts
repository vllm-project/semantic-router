import { afterEach, describe, expect, it, vi } from 'vitest'

import { type EvaluationRequestError, getEvaluationReport } from './evaluationPlaneApi'

const MISSING_RUN_ID = '44444444-4444-4444-8444-444444444444'

afterEach(() => vi.unstubAllGlobals())

describe('Evaluation Plane request errors', () => {
  it('retains the HTTP status needed for bounded recovery decisions', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: { message: 'report missing' } }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    )

    await expect(getEvaluationReport(MISSING_RUN_ID)).rejects.toMatchObject({
      name: 'EvaluationRequestError',
      message: 'report missing',
      status: 404,
    } satisfies Partial<EvaluationRequestError>)
  })
})
