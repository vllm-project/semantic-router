import type {
  BuilderNLProgressEvent,
  BuilderNLReview,
  BuilderNLValidation,
} from '@/types/dsl'
import type { DSLState, DSLStore } from './dslStoreTypes'

export interface DeployStatusService {
  name?: string
  healthy?: boolean
}

export interface DeployStatusResponse {
  overall?: string
  services?: DeployStatusService[]
}

export const initialDSLState: DSLState = {
  dslSource: '',
  yamlOutput: '',
  crdOutput: '',
  diagnostics: [],
  symbols: null,
  ast: null,
  baseConfigYaml: '',
  wasmReady: false,
  wasmError: null,
  loading: false,
  compileError: null,
  mode: 'dsl',
  dirty: false,
  lastCompileAt: null,
  deploying: false,
  deployStep: null,
  deployResult: null,
  showDeployConfirm: false,
  configVersions: [],
  deployPreviewCurrent: '',
  deployPreviewMerged: '',
  deployPreviewLoading: false,
  deployPreviewError: null,
  nlGenerating: false,
  nlGenerateError: null,
  nlStagedDraft: null,
  nlProgressEvents: [],
}

type DSLStoreSetter = (
  partial: Partial<DSLStore> | ((state: DSLStore) => Partial<DSLStore>),
) => void

export function normalizeBuilderNLReview(review: BuilderNLReview | undefined): BuilderNLReview {
  return {
    ready: review?.ready ?? false,
    summary: review?.summary ?? '',
    warnings: Array.isArray(review?.warnings) ? review.warnings : [],
    checks: Array.isArray(review?.checks) ? review.checks : [],
  }
}

export function normalizeBuilderNLValidation(
  validation: BuilderNLValidation | undefined,
): BuilderNLValidation {
  return {
    ready: validation?.ready ?? false,
    diagnostics: Array.isArray(validation?.diagnostics) ? validation.diagnostics : [],
    errorCount: typeof validation?.errorCount === 'number' ? validation.errorCount : 0,
    compileError: validation?.compileError || undefined,
  }
}

export function appendBuilderNLProgress(set: DSLStoreSetter, event: BuilderNLProgressEvent) {
  console.log(`[builder-nl][${event.phase}] ${event.message}`)
  set((state) => ({
    nlProgressEvents: [...state.nlProgressEvents.slice(-79), event],
  }))
}
