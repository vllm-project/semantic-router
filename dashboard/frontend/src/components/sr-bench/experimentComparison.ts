import { benchApi } from './api'
import { experimentApi, type ExperimentMember } from './experimentApi'
import type { RunChoice } from './types'

const maxPages = 100

async function boundedRead<T>(signal: AbortSignal, read: (signal: AbortSignal) => Promise<T>) {
  const controller = new AbortController()
  const abort = () => controller.abort()
  signal.addEventListener('abort', abort, { once: true })
  if (signal.aborted) abort()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, 60000)
  try {
    return await read(controller.signal)
  } catch (cause) {
    if (timedOut)
      throw new Error(
        'Reading this experiment exceeded 60 seconds. Its comparisons remain unverified. Reload to try again.',
      )
    throw cause
  } finally {
    clearTimeout(timer)
    signal.removeEventListener('abort', abort)
  }
}

// Membership is independent of the paged workspace. Do not infer absence from
// one page or use a partly loaded group as a comparison scope.
export function readExperimentMembers(id: string, signal: AbortSignal) {
  return boundedRead(signal, (bounded) => readMembers(id, bounded))
}

async function readMembers(id: string, signal: AbortSignal) {
  const ids = new Set<string>()
  const members: ExperimentMember[] = []
  let after = 0
  let pages = 0
  let revision: string | undefined
  do {
    signal.throwIfAborted()
    if (++pages > maxPages)
      throw new Error(
        'Experiment membership exceeded 100 pages. It was not loaded completely; comparisons remain disabled.',
      )
    const page = await experimentApi.runs(id, after, signal)
    if (
      page.experiment.id !== id ||
      typeof page.experiment.updated_at !== 'string' ||
      (revision !== undefined && page.experiment.updated_at !== revision) ||
      !Array.isArray(page.members) ||
      page.members.length > 20 ||
      typeof page.has_more !== 'boolean' ||
      (page.has_more && (!Number.isInteger(page.next_cursor) || page.next_cursor! <= after))
    )
      throw new Error('Experiment membership changed or could not be read completely. Reload it.')
    revision = page.experiment.updated_at
    for (const member of page.members) {
      if (typeof member.run_id !== 'string' || !member.run_id || ids.has(member.run_id))
        throw new Error('Experiment membership contains duplicate or invalid runs. Reload it.')
      ids.add(member.run_id)
      members.push(member)
    }
    if (!page.has_more) return members
    after = page.next_cursor!
  } while (!signal.aborted)
  signal.throwIfAborted()
  return members
}

// A globally usable baseline may have only out-of-group candidates. Ask the
// same authoritative options API until a scoped pair is verified or exhausted.
export function verifyExperimentBaselines(
  baselines: RunChoice[],
  members: ReadonlySet<string>,
  signal: AbortSignal,
) {
  return boundedRead(signal, (bounded) => verifyBaselines(baselines, members, bounded))
}

async function verifyBaselines(
  baselines: RunChoice[],
  members: ReadonlySet<string>,
  signal: AbortSignal,
) {
  const ids = new Set<string>()
  let scanLimited = false
  let pages = 0
  for (const baseline of baselines) {
    if (!members.has(baseline.run_id)) continue
    let after: string | undefined
    const cursors = new Set<string>()
    do {
      signal.throwIfAborted()
      if (++pages > maxPages)
        throw new Error(
          'Experiment comparison verification exceeded 100 pages. No incomplete selection will be used.',
        )
      const page = await benchApi.runOptions('comparison', baseline.run_id, after, signal)
      if (
        page.model_requests !== 0 ||
        (page.baseline !== null && page.baseline.run_id !== baseline.run_id) ||
        !Array.isArray(page.options) ||
        page.options.length > 10 ||
        page.options.some(
          (option) =>
            typeof option.run_id !== 'string' ||
            !option.run_id ||
            option.run_id === baseline.run_id,
        ) ||
        new Set(page.options.map((option) => option.run_id)).size !== page.options.length ||
        (page.baseline === null && page.options.length > 0) ||
        typeof page.has_more !== 'boolean' ||
        typeof page.scan_limited !== 'boolean' ||
        (page.has_more && (!page.next_cursor || cursors.has(page.next_cursor)))
      )
        throw new Error('The service returned an inconsistent comparison page. Reload options.')
      scanLimited ||= page.scan_limited
      if (page.options.some((option) => members.has(option.run_id))) {
        ids.add(baseline.run_id)
        break
      }
      if (!page.has_more) break
      after = page.next_cursor!
      cursors.add(after)
    } while (!signal.aborted)
  }
  signal.throwIfAborted()
  return { ids, scanLimited }
}

export function comparisonSelectionReady(
  baseline: string,
  candidates: string[],
  baselines: RunChoice[],
  choices: RunChoice[],
) {
  return (
    !!baseline &&
    baselines.some((item) => item.run_id === baseline) &&
    candidates.length > 0 &&
    new Set(candidates).size === candidates.length &&
    candidates.every((id) => choices.some((item) => item.run_id === id))
  )
}
