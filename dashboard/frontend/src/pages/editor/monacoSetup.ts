import { loader } from '@monaco-editor/react'
import 'monaco-editor/esm/vs/editor/edcore.main.js'
import 'monaco-editor/esm/vs/basic-languages/yaml/yaml.contribution.js'
import * as monaco from 'monaco-editor/esm/vs/editor/editor.api.js'
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker.js?worker'

/**
 * Hands the bundled Monaco to the editor components, so the loader never
 * injects its CDN script, which the Dashboard's `script-src 'self'` blocks.
 * Only the DSL view and the deploy diff import this module, lazily.
 */
export async function installMonaco(onWorkerError: (error: unknown) => void): Promise<void> {
  globalThis.MonacoEnvironment = {
    // No language service runs in a worker: the DSL and the YAML diff only
    // need the editor worker, whatever label Monaco asks for.
    getWorker() {
      const worker = new EditorWorker()
      worker.addEventListener('error', onWorkerError)
      return worker
    },
  }
  loader.config({ monaco })
  await loader.init()
}
