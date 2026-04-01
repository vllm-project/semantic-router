import type {
  ProjectionSignalConfig,
  SignalConfig,
  SignalType,
} from '../types'

export interface ProjectionGroup {
  nodeId: string
  mappingName: string
  sourceScore: string
  method: string
  outputs: SignalConfig[]
  upstreamSignals: Array<{ type: SignalType; name: string }>
}

function isProjectionSignalConfig(config: SignalConfig['config']): config is ProjectionSignalConfig {
  return (
    typeof config === 'object'
    && config !== null
    && typeof (config as ProjectionSignalConfig).mapping === 'string'
    && typeof (config as ProjectionSignalConfig).source === 'string'
  )
}

export function buildProjectionGroups(signals: SignalConfig[]): ProjectionGroup[] {
  const groups = new Map<string, ProjectionGroup>()

  signals.forEach((signal) => {
    if (!isProjectionSignalConfig(signal.config)) {
      return
    }

    const mappingName = signal.config.mapping
    const existing = groups.get(mappingName)
    const upstreamSignals = signal.config.upstreamSignals ?? []

    if (!existing) {
      groups.set(mappingName, {
        nodeId: `projection-group-${mappingName.replace(/[^a-zA-Z0-9]/g, '-')}`,
        mappingName,
        sourceScore: signal.config.source,
        method: signal.config.method,
        outputs: [signal],
        upstreamSignals: [...upstreamSignals],
      })
      return
    }

    existing.outputs.push(signal)
    upstreamSignals.forEach((input) => {
      if (existing.upstreamSignals.some((candidate) => candidate.type === input.type && candidate.name === input.name)) {
        return
      }
      existing.upstreamSignals.push(input)
    })
  })

  return Array.from(groups.values())
}

export function buildProjectionOutputNodeMap(groups: ProjectionGroup[]): Map<string, string> {
  const byOutputName = new Map<string, string>()

  groups.forEach((group) => {
    group.outputs.forEach((signal) => {
      byOutputName.set(signal.name, group.nodeId)
    })
  })

  return byOutputName
}

export function groupProjectionInputsBySignalType(group: ProjectionGroup): Map<SignalType, string[]> {
  const grouped = new Map<SignalType, string[]>()

  group.upstreamSignals.forEach((input) => {
    if (!grouped.has(input.type)) {
      grouped.set(input.type, [])
    }
    grouped.get(input.type)!.push(input.name)
  })

  return grouped
}

export function formatProjectionInputLabel(inputNames: string[]): string {
  if (inputNames.length <= 2) {
    return inputNames.join(', ')
  }
  return `${inputNames.slice(0, 2).join(', ')} +${inputNames.length - 2}`
}

export function formatProjectionOutputLabel(outputNames: string[]): string {
  if (outputNames.length <= 2) {
    return outputNames.join(', ')
  }
  return `${outputNames.slice(0, 2).join(', ')} +${outputNames.length - 2}`
}
