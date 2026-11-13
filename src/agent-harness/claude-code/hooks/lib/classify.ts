import { atom, read, update } from 'claude-code'
import type { EngineInterface } from 'claude-code'

import type { Runtime } from './settings'

const HEALTH_TIMEOUT_MS = 3000

// How long an app waits for an answer before it steps aside: a prompt or a spawn holds Claude up, so it waits less
// than a tool's output or a finished answer.
export const QUICK_ANSWER_MS = 1500
export const ANSWER_MS = 3000
// While the runtime is offline one request a minute goes through to see whether it is back.
const RETRY_MS = 60_000

const isOffline = atom({ plugin: 'vsr', key: 'isOffline' } as const, false)
// When a request last went to a runtime known to be offline.
const lastTry = atom({ plugin: 'vsr', key: 'offlineTry' } as const, 0)
// The runtime answers one request at a time and a request cannot be called back, so one vSR gave up on still holds
// it: a new question waits for those, within its own wait, rather than queue behind them and every one after. The
// state names them; the requests themselves are kept here, and an id with none here (the module reloaded) is done.
const abandoned = atom({ plugin: 'vsr', key: 'abandoned' } as const, [] as number[])
const unanswered = new Map<number, Promise<unknown>>()
let nextRequest = 0

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

export const numberAt = (value: unknown, key: string): number | undefined => {
  const found = isRecord(value) ? value[key] : undefined
  return typeof found === 'number' ? found : undefined
}

// A `choice` answer's probability for one key.
export const probabilityOf = (answer: unknown, key: string) =>
  isRecord(answer) ? numberAt(answer.probabilities, key) : undefined

// Marks a skipped app's record when the runtime was offline, so its row stays quiet and the band speaks for it.
// Why a failed question failed, for its record: the runtime offline, or the request cancelled with its call.
export const failureOf = (result: Classified<unknown>) =>
  result.ok ? {} : { ...(result.offline ? { offline: true as const } : {}), ...(result.cancelled ? { cancelled: true as const } : {}) }

// The runtime's whole reply: `answers` per question, and `spans` for span questions.
export type Reply = { answers: Record<string, unknown>; spans: Record<string, unknown> }

// `offline`: the runtime could not be reached, or was skipped as offline; the band says so once, the rows do not.
export type Classified<T> =
  | { ok: true; value: T; latencyMs: number }
  | { ok: false; why: string; latencyMs: number; offline?: true; cancelled?: true }

// A plain prompt, or the fields of a `pair` or `grounded` state.
export type State = string | Record<string, string>

export const headersOf = (runtime: Runtime): Record<string, string> => ({
  'content-type': 'application/json',
  ...(runtime.token === '' ? {} : { authorization: `Bearer ${runtime.token}` }),
})

// The one way every application asks the Router: POST, race a timeout, parse. Never throws.
export async function classify<T>(
  $: EngineInterface,
  runtime: Runtime,
  state: State,
  questions: Record<string, unknown>,
  timeoutMs: number,
  parse: (reply: Reply) => T | undefined,
): Promise<Classified<T>> {
  const started = await $.clock.now()
  const failed = async (why: string, offline = false, cancelled = false) => ({
    ok: false as const,
    why,
    latencyMs: (await $.clock.now()) - started,
    ...(offline ? { offline: true as const } : {}),
    ...(cancelled ? { cancelled: true as const } : {}),
  })
  if (await read($, isOffline)) {
    if (started - (await read($, lastTry)) < RETRY_MS) return failed('the decision model is offline', true)
    await update($, lastTry, () => started)
  }
  const deadline = started + timeoutMs
  for (const id of await read($, abandoned)) {
    const earlier = unanswered.get(id)
    if (earlier !== undefined) {
      const left = Math.max(0, deadline - (await $.clock.now()))
      const isAnswered = await Promise.race([earlier.then(() => true, () => true), $.clock.sleep(left).then(() => false)])
      if (!isAnswered) return failed(`the decision model was still answering an earlier question after ${timeoutMs / 1000} s`)
    }
    await update($, abandoned, ids => ids.filter(one => one !== id))
  }
  const request = $.http.fetch(runtime.url, { method: 'POST', headers: headersOf(runtime), body: JSON.stringify({ ...(runtime.model === '' ? {} : { model: runtime.model }), state, questions }) })
  let response
  try {
    response = await Promise.race([request, $.clock.sleep(Math.max(0, deadline - (await $.clock.now()))).then(() => undefined)])
  } catch (error) {
    // A request cancelled with its call (the user pressed Esc) says nothing about the runtime: only a fetch that
    // could not connect makes it offline.
    if (isCancelled(error)) return await failed('the request was cancelled', false, true)
    await markOffline($, runtime)
    return await failed(`could not reach the decision model (${String(error).split(': ').pop()})`, true)
  }
  if (response === undefined) {
    // Given up on, it still holds the runtime until it is answered.
    const id = nextRequest++
    const drop = () => unanswered.delete(id)
    unanswered.set(id, request)
    void request.then(drop, drop)
    await update($, abandoned, ids => [...ids, id])
    // Slow or gone: /health tells, without holding this call up.
    void checkRuntime($, runtime)
    return await failed(`the decision model took over ${timeoutMs / 1000} s`)
  }
  if (!response.ok) return await failed(`the decision model answered with an error (HTTP ${response.status})`)
  // Something answered: a body that is not the runtime's (a proxy's sign-in page) fails this question, it does not
  // make the runtime offline.
  const value = valueOf(response.text, parse)
  if (value === undefined) return await failed('the decision model gave an answer vSR could not read')
  await markReachable($)
  return { ok: true, value, latencyMs: (await $.clock.now()) - started }
}

const isCancelled = (error: unknown) =>
  (error instanceof Error && error.name === 'AbortError') || /\boperation was aborted\b|\baborted\b/i.test(String(error))

// The parsed answer of a reply's text, or undefined for a text that is not JSON or not of the runtime's shape.
function valueOf<T>(text: string, parse: (reply: Reply) => T | undefined): T | undefined {
  try {
    const body: unknown = JSON.parse(text)
    const answers = isRecord(body) ? body.answers : undefined
    const spans = isRecord(body) && isRecord(body.spans) ? body.spans : {}
    return isRecord(answers) ? parse({ answers, spans }) : undefined
  } catch {
    return undefined
  }
}

// Read while drawing, it redraws the band when the runtime goes offline or comes back.
export async function readOffline($: EngineInterface): Promise<boolean> {
  return read($, isOffline)
}

async function markReachable($: EngineInterface) {
  if (await read($, isOffline)) await update($, isOffline, () => false)
}

// The runtime's base URL (its `/v1/decisions` left off), and `/health` or `/v1/models` after it.
const baseOf = (url: string) => url.replace(/\/v1\/decisions\/?$/, '').replace(/\/$/, '')
export const healthUrlOf = (url: string) => `${baseOf(url)}/health`
export const modelsUrlOf = (url: string) => `${baseOf(url)}/v1/models`
// A System One API's listing: its models, and the listener's vsr profile beside them.
export const listingUrlOf = (url: string) => `${baseOf(url)}/v1/systemone/models`
// The runtime's model and the question types it answers, from a System One API's model list or the one model
// vllm-srun lists; undefined when neither names it with its types.
export async function modelOf($: EngineInterface, runtime: Runtime): Promise<{ id: string; types: string[] } | undefined> {
  for (const url of [listingUrlOf(runtime.url), modelsUrlOf(runtime.url)]) {
    const text = await $.http.fetch(url, { headers: headersOf(runtime) }).then(r => (r.ok ? r.text : ''), () => '')
    let data: unknown
    try {
      data = (JSON.parse(text) as { data?: unknown }).data
    } catch {
      continue
    }
    const models = Array.isArray(data) ? data.filter(isRecord) : []
    const model = runtime.model === '' ? models[0] : models.find(one => one.id === runtime.model)
    if (model !== undefined && typeof model.id === 'string' && Array.isArray(model.question_types)) {
      return { id: model.id, types: model.question_types.filter(type => typeof type === 'string') }
    }
  }
  return undefined
}
// A `/v1/models` answer that lists at least one model, as the setup script's health check requires.
const listsModels = (response: { ok: boolean; text: string }) => {
  if (!response.ok) return false
  try {
    const data = (JSON.parse(response.text) as { data?: unknown }).data
    return Array.isArray(data) && data.length > 0
  } catch {
    return false
  }
}

// Offline from now: requests are skipped for a minute at a time, and a toast says so once.
async function markOffline($: EngineInterface, runtime: Runtime) {
  const now = await $.clock.now()
  await update($, lastTry, () => now)
  if (await read($, isOffline)) return
  await update($, isOffline, () => true)
  $.ui.toast(`vSR: the decision model is not answering at ${runtime.url} · vSR's checks step aside until it does (apps set to wait until done hold instead); guard's fixed rules still hold · /vsr:setup`)
}

// Asks the runtime's `/health` once; one that does not answer in time is offline. A hosted System One API has no
// `/health`, so a 404 there is answered by its `/v1/models` instead. True when it answered.
export async function checkRuntime($: EngineInterface, runtime: Runtime): Promise<boolean> {
  const answered = await Promise.race([
    $.http.fetch(healthUrlOf(runtime.url), { headers: headersOf(runtime) }).then(
      response =>
        response.ok ||
        (response.status === 404 && $.http.fetch(modelsUrlOf(runtime.url), { headers: headersOf(runtime) }).then(listsModels, () => false)),
      () => false,
    ),
    $.clock.sleep(HEALTH_TIMEOUT_MS).then(() => false),
  ])
  await (answered ? markReachable($) : markOffline($, runtime))
  return answered
}
