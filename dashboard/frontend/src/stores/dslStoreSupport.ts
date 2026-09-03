import type { DSLState } from './dslStoreTypes'

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
  renderedYamlOutput: '',
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
  mode: 'visual',
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
}
