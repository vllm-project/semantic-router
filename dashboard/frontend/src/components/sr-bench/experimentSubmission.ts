export interface PendingExperiment {
  version: 1
  actorID: string
  name: string
  key: string
}
export const experimentKey = (actorID: string) =>
  `sr-bench-experiment:${encodeURIComponent(actorID)}`
export function readExperiment(actorID: string): {
  saved: PendingExperiment | null
  error: string
} {
  try {
    if (!actorID) throw new Error('Account required')
    const raw = sessionStorage.getItem(experimentKey(actorID))
    if (!raw) return { saved: null, error: '' }
    if (raw.length > 16384) throw new Error('Oversized request')
    const saved = JSON.parse(raw) as PendingExperiment
    if (
      saved.version !== 1 ||
      saved.actorID !== actorID ||
      typeof saved.name !== 'string' ||
      !saved.name.trim() ||
      saved.name.length > 160 ||
      typeof saved.key !== 'string' ||
      !saved.key.trim()
    )
      throw new Error('Invalid identity')
    return { saved, error: '' }
  } catch {
    return {
      saved: null,
      error:
        'The saved experiment submission cannot be read safely. Reconcile the experiment list before clearing this tab’s saved request.',
    }
  }
}
export function saveExperiment(saved: PendingExperiment) {
  const current = readExperiment(saved.actorID)
  if (current.error || (current.saved && JSON.stringify(current.saved) !== JSON.stringify(saved)))
    throw new Error('Another experiment submission needs reconciliation. No request was sent.')
  try {
    sessionStorage.setItem(experimentKey(saved.actorID), JSON.stringify(saved))
  } catch {
    throw new Error(
      'The browser could not preserve this experiment submission. No request was sent.',
    )
  }
}
export function clearExperiment(saved: PendingExperiment): boolean {
  const current = readExperiment(saved.actorID)
  if (current.error || JSON.stringify(current.saved) !== JSON.stringify(saved)) return false
  sessionStorage.removeItem(experimentKey(saved.actorID))
  return true
}
