import { DEFAULT_DECISION_MODEL, type DecisionModelOption } from './decisionModelSupport'

/** The CPU recommendation order: the first one the runtime serves now is recommended. */
export const CPU_RECOMMENDED_DECISION_MODELS = [
  'vela3.1-fast',
  'vela3.1-pico-150m',
  'vela3.1-pico-380m',
] as const

const ACCELERATOR = /^(cuda|rocm|xpu|mps|npu|hpu)(:|$)/

export interface DecisionHardwareObservation {
  /** The Router's report that a prepared model runs on a GPU. */
  gpuAvailable?: boolean
  /** Devices the running instance reports: runtime deployments, replicas and loaded models. */
  observedDevices: readonly (string | undefined)[]
  /** Devices the saved decision deployments name; consulted only when nothing is observed. */
  configuredDevices: readonly (string | undefined)[]
}

function devices(values: readonly (string | undefined)[]): string[] {
  return values
    .map((value) => value?.trim().toLowerCase() ?? '')
    .filter((value) => value && value !== 'auto')
}

/**
 * The model to mark "Recommended for CPU": none unless the instance runs
 * decision models on a CPU only; then the first model of the CPU order that
 * the catalog can serve, else the CPU default.
 */
export function cpuRecommendedDecisionModel(
  options: readonly Pick<DecisionModelOption, 'name' | 'servable'>[],
  hardware: DecisionHardwareObservation,
): string | null {
  const observed = devices(hardware.observedDevices)
  const evidence = observed.length ? observed : devices(hardware.configuredDevices)
  const cpuOnly =
    hardware.gpuAvailable !== true &&
    evidence.includes('cpu') &&
    !evidence.some((device) => ACCELERATOR.test(device))
  if (!cpuOnly) return null
  const servable = new Set(options.filter((option) => option.servable).map((option) => option.name))
  return (
    CPU_RECOMMENDED_DECISION_MODELS.find((name) => servable.has(name)) ??
    (servable.has(DEFAULT_DECISION_MODEL) ? DEFAULT_DECISION_MODEL : null)
  )
}
