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
  baseConfigFromRouter: false,
  compilerReady: false,
  compilerError: null,
  loading: false,
  compileError: null,
  mode: 'visual',
  dirty: false,
  savedSource: '',
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

// A base loaded from the Router is a copy of the live config that goes stale
// when another page saves, so a deploy and its rendered YAML apply the draft to
// the live config. An imported config is sent as the base that replaces it.
export function deployBaseYaml(
  state: Pick<DSLState, 'baseConfigYaml' | 'baseConfigFromRouter'>,
): string {
  return state.baseConfigFromRouter ? '' : state.baseConfigYaml
}
