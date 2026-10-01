import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { SrBenchRequestError } from './api'
import { datasetPreparationApi, type DatasetPreparationJob } from './datasetPreparationApi'
import { prepareEvaluationDataset } from './evaluationPreparation'

const body = { benchmarks: ['mmlu-pro', 'gpqa-diamond'], profile: 'quick' as const, seed: 42 }
const dataset = {
  id: 'd'.repeat(64),
  path: '/store/dataset.jsonl',
  sha256: 'c'.repeat(64),
  case_count: 540,
  profile: 'quick',
  seed: 42,
  benchmarks: ['gpqa-diamond', 'mmlu-pro'],
}
const job = (patch: Partial<DatasetPreparationJob> = {}): DatasetPreparationJob => ({
  id: `prep-${'a'.repeat(32)}`,
  benchmarks: dataset.benchmarks,
  profile: 'quick',
  seed: 42,
  status: 'completed',
  phase: 'completed',
  dataset,
  created_at: '2026-09-21T00:00:00Z',
  updated_at: '2026-09-21T00:00:00Z',
  ...patch,
})

beforeEach(() => {
  vi.spyOn(datasetPreparationApi, 'list').mockResolvedValue({ preparations: [] })
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('evaluation prerequisite preparation', () => {
  it('observes the durable batch until its exact frozen dataset is ready', async () => {
    vi.useFakeTimers()
    const post = vi.spyOn(datasetPreparationApi, 'prepare').mockResolvedValue({
      preparation: job({ status: 'running', phase: 'installing_dependencies', dataset: undefined }),
    })
    const get = vi.spyOn(datasetPreparationApi, 'get').mockResolvedValue({ preparation: job() })
    const progress = vi.fn()
    const pending = prepareEvaluationDataset(body, new AbortController().signal, progress)
    await vi.advanceTimersByTimeAsync(1500)
    await expect(pending).resolves.toEqual(dataset)
    expect(post).toHaveBeenCalledExactlyOnceWith(
      { ...body, benchmarks: dataset.benchmarks },
      expect.any(AbortSignal),
    )
    expect(get).toHaveBeenCalledOnce()
    expect(progress.mock.calls.map(([value]) => value.phase)).toEqual([
      'resolving_datasets',
      'installing_dependencies',
      'completed',
    ])
  })

  it('reconciles a lost POST response by reading the matching completed job', async () => {
    const post = vi
      .spyOn(datasetPreparationApi, 'prepare')
      .mockRejectedValue(new TypeError('Lost response'))
    vi.mocked(datasetPreparationApi.list)
      .mockResolvedValueOnce({ preparations: [] })
      .mockResolvedValue({ preparations: [job()] })
    await expect(
      prepareEvaluationDataset(body, new AbortController().signal, vi.fn()),
    ).resolves.toEqual(dataset)
    expect(post).toHaveBeenCalledOnce()
  })

  it('does not hide an access denial by attaching to another preparation', async () => {
    vi.spyOn(datasetPreparationApi, 'prepare').mockRejectedValue(
      new SrBenchRequestError('Forbidden', 403),
    )
    const list = vi.spyOn(datasetPreparationApi, 'list')
    await expect(
      prepareEvaluationDataset(body, new AbortController().signal, vi.fn()),
    ).rejects.toThrow('Forbidden')
    expect(list).toHaveBeenCalledOnce()
  })

  it('does not repeat an ambiguous submission when no matching job is visible', async () => {
    const post = vi
      .spyOn(datasetPreparationApi, 'prepare')
      .mockRejectedValue(new TypeError('Lost response'))
    vi.spyOn(datasetPreparationApi, 'list').mockResolvedValue({ preparations: [] })
    await expect(
      prepareEvaluationDataset(body, new AbortController().signal, vi.fn()),
    ).rejects.toThrow('Lost response')
    expect(post).toHaveBeenCalledOnce()
  })

  it('waits and retries only when the service explicitly rejected a busy slot', async () => {
    vi.useFakeTimers()
    const post = vi
      .spyOn(datasetPreparationApi, 'prepare')
      .mockRejectedValueOnce(new SrBenchRequestError('Busy', 409, 'preparation_busy'))
      .mockResolvedValue({ preparation: job() })
    const progress = vi.fn()
    const pending = prepareEvaluationDataset(body, new AbortController().signal, progress)
    await vi.advanceTimersByTimeAsync(1500)
    await expect(pending).resolves.toEqual(dataset)
    expect(post).toHaveBeenCalledTimes(2)
    expect(progress).toHaveBeenCalledWith({ phase: 'waiting_for_slot' })
  })

  it('leaving the page stops observation without cancelling or resubmitting service work', async () => {
    vi.useFakeTimers()
    const controller = new AbortController()
    const post = vi.spyOn(datasetPreparationApi, 'prepare').mockResolvedValue({
      preparation: job({ status: 'running', phase: 'downloading', dataset: undefined }),
    })
    const get = vi.spyOn(datasetPreparationApi, 'get')
    const pending = prepareEvaluationDataset(body, controller.signal, vi.fn())
    const assertion = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
    await vi.advanceTimersByTimeAsync(0)
    controller.abort()
    await assertion
    await vi.advanceTimersByTimeAsync(5000)
    expect(post).toHaveBeenCalledOnce()
    expect(get).not.toHaveBeenCalled()
  })

  it('requires explicit retry when the preparation worker is unavailable', async () => {
    const post = vi
      .spyOn(datasetPreparationApi, 'prepare')
      .mockRejectedValue(
        new SrBenchRequestError('Retry explicitly.', 503, 'preparation_unavailable'),
      )
    const list = vi.spyOn(datasetPreparationApi, 'list')
    await expect(
      prepareEvaluationDataset(body, new AbortController().signal, vi.fn()),
    ).rejects.toThrow('Retry explicitly.')
    expect(post).toHaveBeenCalledOnce()
    expect(list).toHaveBeenCalledOnce()
  })

  it('does not automatically retry a busy code without its definitive rejection status', async () => {
    const post = vi
      .spyOn(datasetPreparationApi, 'prepare')
      .mockRejectedValue(new SrBenchRequestError('Unavailable', 503, 'preparation_busy'))
    const list = vi.spyOn(datasetPreparationApi, 'list').mockResolvedValue({ preparations: [] })
    await expect(
      prepareEvaluationDataset(body, new AbortController().signal, vi.fn()),
    ).rejects.toThrow('Unavailable')
    expect(post).toHaveBeenCalledOnce()
    expect(list).toHaveBeenCalledTimes(2)
  })

  it('does not recover a lost response using a completed batch from before this request', async () => {
    const post = vi
      .spyOn(datasetPreparationApi, 'prepare')
      .mockRejectedValue(new TypeError('Lost response'))
    vi.mocked(datasetPreparationApi.list).mockResolvedValue({ preparations: [job()] })
    await expect(
      prepareEvaluationDataset(body, new AbortController().signal, vi.fn()),
    ).rejects.toThrow('Lost response')
    expect(post).toHaveBeenCalledOnce()
  })

  it('surfaces a newly failed preparation without falling back to older success', async () => {
    const post = vi
      .spyOn(datasetPreparationApi, 'prepare')
      .mockRejectedValue(new TypeError('Lost response'))
    const failure = job({
      id: `prep-${'b'.repeat(32)}`,
      status: 'failed',
      phase: 'failed',
      error: 'Conflicting sources',
    })
    vi.mocked(datasetPreparationApi.list)
      .mockResolvedValueOnce({ preparations: [job()] })
      .mockResolvedValue({ preparations: [failure, job()] })
    await expect(
      prepareEvaluationDataset(body, new AbortController().signal, vi.fn()),
    ).rejects.toThrow('Conflicting sources')
    expect(post).toHaveBeenCalledOnce()
  })

  it.each([
    job({ seed: 43 }),
    job({ dataset: { ...dataset, benchmarks: ['mmlu-pro'] } }),
    job({ dataset: { ...dataset, profile: 'standard' } }),
  ])('rejects a different scope or frozen dataset', async (preparation) => {
    vi.spyOn(datasetPreparationApi, 'prepare').mockResolvedValue({ preparation })
    await expect(
      prepareEvaluationDataset(body, new AbortController().signal, vi.fn()),
    ).rejects.toThrow(/matches|differs/)
  })
})
