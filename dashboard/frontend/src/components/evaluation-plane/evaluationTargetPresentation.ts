import type { EvaluationCatalogTarget } from '../../types/evaluationPlane'

type TargetPresentationIdentity = Pick<EvaluationCatalogTarget, 'id' | 'name'>

export function targetPresentationLabel(target: TargetPresentationIdentity): string {
  return target.name
}

export function targetOptionLabels(
  targets: readonly TargetPresentationIdentity[],
): Map<string, string> {
  const distinctTargets = [...new Map(targets.map((target) => [target.id, target])).values()]
  const targetsByName = new Map<string, TargetPresentationIdentity[]>()
  for (const target of distinctTargets) {
    const label = targetPresentationLabel(target)
    targetsByName.set(label, [...(targetsByName.get(label) || []), target])
  }

  const labels = new Map<string, string>()
  for (const [label, sameNameTargets] of targetsByName) {
    if (sameNameTargets.length === 1) {
      labels.set(sameNameTargets[0].id, label)
      continue
    }
    const sortedTargets = [...sameNameTargets].sort((left, right) =>
      left.id.localeCompare(right.id),
    )
    sortedTargets.forEach((target, index) =>
      labels.set(target.id, `${label} · Option ${index + 1}`),
    )
  }
  return labels
}
