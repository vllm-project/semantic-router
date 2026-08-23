import type { DSLState } from './dslStoreTypes'

export const initialDSLState: DSLState = {
  dslSource: '',
  yamlOutput: '',
  crdOutput: '',
  diagnostics: [],
  symbols: null,
  ast: null,
  wasmReady: false,
  wasmError: null,
  loading: false,
  compileError: null,
  mode: 'visual',
  dirty: false,
  lastCompileAt: null,
}
