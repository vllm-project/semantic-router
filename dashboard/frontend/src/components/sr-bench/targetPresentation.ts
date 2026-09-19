import type { Manifest, Target } from './types'

export function targetLabel(target: Pick<Target, 'model'> | null | undefined): string {
  return target?.model.trim() ? target.model : 'Unknown model'
}

export function targetName(
  manifest: Pick<Manifest, 'targets' | 'auxiliary_targets'> | null | undefined,
  id: string,
): string {
  return targetLabel(
    manifest?.targets.find((target) => target.id === id) ?? manifest?.auxiliary_targets?.[id],
  )
}
