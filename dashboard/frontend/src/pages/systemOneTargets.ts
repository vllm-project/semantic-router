import { systemOneDeploymentOption } from './systemOneDeploymentPresentation'

export interface SystemOneDeployment {
  id: string
  model: string
  ready: boolean
  question_types: string[]
  surfaces: string[]
  unavailable_reason?: string
  repo?: string
  family?: string
  max_input_tokens?: number
  max_scan_tokens?: number
}
export interface SystemOneRoute {
  model: string
  recipe: string
  algorithms: string[]
  question_types: string[]
  timeout_ms: number
}
export interface SystemOneRoutes {
  available: boolean
  routes: SystemOneRoute[]
}
export interface SystemOneTarget extends SystemOneDeployment {
  key: string
  kind: 'route' | 'deployment'
  timeout_ms: number
  recipe?: string
  algorithms?: string[]
}

export function isSystemOneRoutes(value: unknown): value is SystemOneRoutes {
  if (!value || typeof value !== 'object') return false
  const body = value as SystemOneRoutes
  return (
    typeof body.available === 'boolean' &&
    Array.isArray(body.routes) &&
    body.routes.every(
      (route) =>
        route &&
        typeof route.model === 'string' &&
        !!route.model &&
        typeof route.recipe === 'string' &&
        Array.isArray(route.algorithms) &&
        route.algorithms.every((algorithm) => typeof algorithm === 'string') &&
        Array.isArray(route.question_types) &&
        route.question_types.every((type) => typeof type === 'string') &&
        Number.isFinite(route.timeout_ms) &&
        route.timeout_ms >= 0,
    )
  )
}

export function systemOneTargets(
  deployments: SystemOneDeployment[],
  routes: SystemOneRoutes | null,
  directTimeout = 30000,
): SystemOneTarget[] {
  return [
    ...(routes?.available
      ? routes.routes.map(
          (route): SystemOneTarget => ({
            ...route,
            id: route.model,
            key: `route:${route.model}`,
            kind: 'route',
            ready: true,
            surfaces: ['decisions'],
          }),
        )
      : []),
    ...deployments.map(
      (deployment): SystemOneTarget => ({
        ...deployment,
        key: `deployment:${deployment.id}`,
        kind: 'deployment',
        timeout_ms: directTimeout,
      }),
    ),
  ]
}

export function systemOneTargetOptions(targets: SystemOneTarget[]) {
  return targets.map((target) =>
    target.kind === 'route'
      ? {
          value: target.key,
          label: target.model,
          group: 'Automatic routes',
          description: `${target.recipe} · ${target.algorithms?.join(' / ')} · ${target.question_types.join(', ')}`,
        }
      : {
          ...systemOneDeploymentOption(target),
          value: target.key,
          group: 'Direct models',
        },
  )
}

export function systemOneTargetRequest(target: SystemOneTarget, request: unknown) {
  return target.kind === 'route'
    ? { path: '/api/decision-model/routes', body: { model: target.id, request } }
    : { path: '/api/decision-model/test', body: { deployment: target.id, request } }
}

export function systemOneTargetTimeout(target: SystemOneTarget): number {
  // Leave transport grace after the Router's own recipe deadline. Browsers
  // cap setTimeout at a signed 32-bit delay; never overflow to an instant abort.
  return Math.min(2147483647, Math.max(1000, target.timeout_ms) + 10000)
}
