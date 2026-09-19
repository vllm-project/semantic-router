import type { RunEvent } from './types'

const lifecycle: Record<string, [string, string]> = {
  created: ['Evaluation created', 'The frozen plan was saved.'],
  running: ['Evaluation started', 'The worker began processing this run.'],
  completed: ['Evaluation completed', 'The run finished. Review Results for quality and cost.'],
  failed: ['Evaluation failed', 'The run stopped after a failure.'],
  interrupted: [
    'Evaluation interrupted',
    'Execution stopped before the run finished. Review saved evidence before recovery.',
  ],
  cancelled: ['Evaluation cancelled', 'Cancellation finished; saved results remain available.'],
  cancellation_requested: [
    'Cancellation requested',
    'A stop was requested. This event alone does not confirm that in-flight requests stopped.',
  ],
  accounting_reconciled: [
    'Accounting reconciled',
    'Saved usage was reconciled without new model requests. Original receipts remain unchanged.',
  ],
  failure_observed: ['Failure recorded', 'The worker saved a failure for review.'],
  call_sent: [
    'Model request dispatched',
    'A request was sent. This event alone does not confirm completion or known cost.',
  ],
  call_replayed: [
    'Saved answer reused',
    'A previous answer was reused for diagnostic replay; no new model generation.',
  ],
  case_completed: [
    'Case result saved',
    'Processing finished. Completion does not imply a correct answer.',
  ],
  case_failed: ['Case failed', 'A failed case result was saved.'],
  case_cancelled: ['Case cancelled', 'A cancelled case result was saved.'],
}

export function describeRunEvent(event: RunEvent) {
  const kind = String(event.kind ?? event.type ?? event.event ?? 'event')
  const data =
    event.data && typeof event.data === 'object' && !Array.isArray(event.data)
      ? (event.data as Record<string, unknown>)
      : event
  const text = (key: string) => (typeof data[key] === 'string' ? (data[key] as string) : undefined)
  const [title, description] = lifecycle[kind] ?? [
    kind.replace(/[_-]+/g, ' ').replace(/^./, (letter) => letter.toUpperCase()),
    'A worker event was recorded.',
  ]
  const group = kind.startsWith('case_') ? 'cases' : kind.startsWith('call_') ? 'requests' : 'run'
  return {
    title,
    description,
    group,
    timestamp: event.at ?? event.timestamp,
    caseID: text('case_id'),
    targetID: text('target_id'),
    callID: text('call_id'),
    role: text('role'),
    reason: text('reason') ?? text('error'),
    attention: ['failed', 'interrupted', 'failure_observed', 'case_failed'].includes(kind),
  }
}
