import type { Target } from './types'

/** Registry capability checks for the picker; the service revalidates the frozen plan and recipe. */
export function nativeOutputIssue(target: Target): string | null {
  const entries = Object.entries(target.native_limits ?? {})
  if (!entries.length)
    return 'The operator has not registered native context and output limits for this target.'
  if (
    entries.some(
      ([model, limits]) =>
        !model ||
        !limits ||
        !Number.isSafeInteger(limits.context_window) ||
        !Number.isSafeInteger(limits.max_output_tokens) ||
        limits.context_window <= 0 ||
        limits.max_output_tokens <= 0 ||
        limits.max_output_tokens > limits.context_window,
    )
  )
    return 'The registered native context and output limits are incomplete or invalid.'
  if (
    target.kind === 'single' &&
    (entries.length !== 1 || entries[0][0] !== (target.expected_response_model || target.model))
  )
    return 'The registered native limits do not identify this model.'
  if (Object.prototype.hasOwnProperty.call(target.request_params ?? {}, 'max_tokens'))
    return 'This target has a fixed output cap. Choose a registered profile without that cap.'
  if (target.kind === 'mom' && (target.max_inference_calls !== 1 || target.capture_recipe !== true))
    return 'Native capacity requires a recipe with one dispatch and verified recipe capture.'
  return null
}
