import type { SystemOneDeployment } from './useSystemOnePlayground'

export function systemOneDeploymentOption(deployment: SystemOneDeployment) {
  const label = deployment.repo?.trim() || deployment.model || deployment.id
  return {
    value: deployment.id,
    label,
    description: [
      ...(label === deployment.id ? [] : [`Deployment: ${deployment.id}`]),
      deployment.ready ? 'Ready' : 'Unavailable',
      `${deployment.question_types.length} question types`,
    ].join(' · '),
  }
}
