import type { ProjectionMapping, ProjectionPartition, ProjectionScore } from './configPageSupport'

function includesQuery(values: Array<string | undefined>, query: string): boolean {
  return values.join(' ').toLocaleLowerCase().includes(query.trim().toLocaleLowerCase())
}

export function matchesProjectionPartition(partition: ProjectionPartition, query: string): boolean {
  return includesQuery(
    [partition.name, partition.semantics, partition.default, ...(partition.members || [])],
    query,
  )
}

export function matchesProjectionScore(score: ProjectionScore, query: string): boolean {
  return includesQuery(
    [
      score.name,
      score.method,
      ...(score.inputs || []).flatMap((input) => [input.type, input.name, input.value_source]),
    ],
    query,
  )
}

export function matchesProjectionMapping(mapping: ProjectionMapping, query: string): boolean {
  return includesQuery(
    [
      mapping.name,
      mapping.source,
      mapping.method,
      mapping.calibration?.method,
      ...(mapping.outputs || []).map((output) => output.name),
    ],
    query,
  )
}
