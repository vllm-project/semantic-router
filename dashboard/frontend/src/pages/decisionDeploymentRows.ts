import type { CanonicalModelDeployment } from './configPageSupport'
import type { RouterConfig } from './dashboardPageTypes'
import {
  decisionDeploymentProgress,
  decisionRuntimeModelName,
  type DecisionDeploymentProgress,
  type DecisionModelActivation,
  type ModelRuntimeDeployment,
  type ModelRuntimeInventory,
} from './decisionModelManagement'
import {
  DEFAULT_DECISION_MODEL,
  DECISION_MODEL_OPTIONS,
  configuredDecisionDeployment,
  decisionModelByArtifact,
  type DecisionModelOption,
} from './decisionModelSupport'
import { deploymentPlacements } from './decisionReplicaConfig'
import {
  configuredDecisionRuntimes,
  decisionRuntimeConsumers,
  decisionRuntimeDeclarations,
  type DecisionRuntimeConsumer,
} from './decisionRuntimeDeployment'
import type { DecisionTaskBinding, DecisionTasks } from './useDecisionTasks'

export interface DecisionDeploymentRow {
  name: string
  kind: 'model_runtime' | 'systemone' | 'observed'
  /** The authored declaration; only an authored model_runtime declaration is editable here. */
  declaration?: CanonicalModelDeployment
  observed?: ModelRuntimeDeployment
  model?: DecisionModelOption
  modelName: string
  artifact?: string
  device: string
  isDefault: boolean
  consumers: DecisionRuntimeConsumer[]
  /** Questions and selectors that name no deployment; the default deployment answers them. */
  inherited: DecisionRuntimeConsumer[]
  taskBindings: DecisionTaskBinding[]
  progress: DecisionDeploymentProgress
  replicas: { ready: number | null; desired: number }
  /** What it answers, when the catalog, the Router or its declaration says; null is unknown. */
  questionTypes: readonly string[] | null
}

function placementSummary(declaration: CanonicalModelDeployment | undefined): string {
  if (!declaration) return ''
  if (declaration.provider === 'systemone') return 'System One API'
  const placements = deploymentPlacements(declaration)
  const devices = placements.map((placement) =>
    placement.endpoint !== undefined ? 'attached' : placement.device || 'auto',
  )
  const unique = [...new Set(devices)]
  return unique.length === 1 && devices.length > 1
    ? `${unique[0]} ×${devices.length}`
    : unique.join(' + ')
}

export function decisionDeploymentRows({
  config,
  effectiveDeployments,
  inventory,
  activation,
  tasks,
}: {
  config: RouterConfig | null
  /** The effective model catalog, which pre-declares deployments the authored one omits. */
  effectiveDeployments?: Record<string, CanonicalModelDeployment>
  inventory: ModelRuntimeInventory | null
  activation: DecisionModelActivation | null
  tasks: DecisionTasks | null
}): DecisionDeploymentRow[] {
  const authored = decisionRuntimeDeclarations(config)
  const consumers = decisionRuntimeConsumers(config)
  const defaultName = config
    ? configuredDecisionDeployment(config)
    : tasks?.default_deployment || 'primary'
  const names = new Set<string>([
    defaultName,
    ...configuredDecisionRuntimes(config, inventory).map(([name]) => name),
    ...Object.entries(authored)
      .filter(([, declaration]) => declaration.provider === 'systemone')
      .map(([name]) => name),
    ...(inventory?.deployments ?? [])
      .filter((deployment) => deployment.surfaces?.includes('question'))
      .map((deployment) => deployment.name),
  ])
  return [...names]
    .map((name): DecisionDeploymentRow => {
      const declaration = authored[name]
      const effective = declaration ?? effectiveDeployments?.[name]
      const remote = effective?.provider === 'systemone'
      const routerView = tasks?.deployments.find((item) => item.deployment === name)
      const observed =
        inventory?.deployments.find((item) => item.name === name) ??
        (remote && routerView
          ? {
              name,
              managed: false,
              served_name: effective?.model ?? routerView.model,
              ready: routerView.ready,
              state: routerView.ready ? 'ready' : 'not ready',
            }
          : undefined)
      const isDefault = name === defaultName
      const artifact = effective?.artifact ?? observed?.repo
      const model =
        decisionModelByArtifact(artifact) ??
        (isDefault && !effective && !observed
          ? DECISION_MODEL_OPTIONS.find((option) => option.name === DEFAULT_DECISION_MODEL)
          : undefined)
      const own = consumers.filter((consumer) => consumer.deployment === name)
      const inherited = isDefault ? consumers.filter((consumer) => !consumer.deployment) : []
      const selectors = new Set(
        own.filter((consumer) => consumer.kind === 'selector').map((consumer) => `selector.${consumer.name}`),
      )
      const taskBindings =
        tasks?.bindings.filter(
          (binding) => binding.deployment === name && !selectors.has(binding.consumer),
        ) ?? []
      const bound = isDefault || own.length > 0 || taskBindings.length > 0
      const placements = effective && !remote ? deploymentPlacements(effective) : []
      return {
        name,
        kind: remote ? 'systemone' : effective || !observed ? 'model_runtime' : 'observed',
        declaration,
        observed,
        model,
        modelName:
          model?.name ??
          (remote ? effective?.model : undefined) ??
          (observed ? decisionRuntimeModelName(observed) : undefined) ??
          artifact?.split('/').slice(-1)[0] ??
          name,
        artifact: artifact ?? model?.artifact,
        device: observed?.device || placementSummary(effective) || (effective ? 'auto' : ''),
        isDefault,
        consumers: own,
        inherited,
        taskBindings,
        progress: decisionDeploymentProgress({
          declared: Boolean(effective) || isDefault,
          bound,
          activation,
          observed,
        }),
        replicas: observed?.desired_replicas
          ? { ready: observed.ready_replicas ?? 0, desired: observed.desired_replicas }
          : observed
            ? { ready: observed.ready ? 1 : 0, desired: Math.max(1, placements.length) }
            : { ready: null, desired: Math.max(1, placements.length) },
        questionTypes:
          model?.questionTypes ??
          routerView?.native_question_types ??
          effective?.capabilities?.question_types ??
          null,
      }
    })
    .sort(
      (left, right) =>
        Number(right.isDefault) - Number(left.isDefault) || left.name.localeCompare(right.name),
    )
}

export function decisionConsumerSummary(row: DecisionDeploymentRow): string {
  const questions = row.consumers.filter((consumer) => consumer.kind === 'question').length
  const selectors = row.consumers.filter((consumer) => consumer.kind === 'selector').length
  const parts = [
    row.isDefault ? 'default' : '',
    questions ? `${questions} ${questions === 1 ? 'signal' : 'signals'}` : '',
    selectors ? `${selectors} ${selectors === 1 ? 'decision' : 'decisions'}` : '',
    row.taskBindings.length
      ? `${row.taskBindings.length} ${row.taskBindings.length === 1 ? 'task' : 'tasks'}`
      : '',
  ].filter(Boolean)
  return parts.length ? parts.join(' · ') : 'Not bound'
}

/** Devices the running instance reports for its decision deployments. */
export function observedDecisionDevices(rows: DecisionDeploymentRow[]): string[] {
  return rows.flatMap((row) => [
    row.observed?.device ?? '',
    ...(row.observed?.replicas ?? []).map((replica) => replica.device ?? ''),
  ])
}

/** Devices the saved decision deployments name. */
export function configuredDecisionDevices(rows: DecisionDeploymentRow[]): string[] {
  return rows.flatMap((row) =>
    row.declaration && row.kind === 'model_runtime'
      ? deploymentPlacements(row.declaration).map((placement) => placement.device ?? '')
      : [],
  )
}
