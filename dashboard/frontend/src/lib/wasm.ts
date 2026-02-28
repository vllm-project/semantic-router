/**
 * WASM Bridge — loads the Go-compiled signal-compiler.wasm and
 * exposes typed wrappers around the global JS functions:
 *   signalCompile, signalValidate, signalDecompile, signalFormat
 *
 * Performance optimizations:
 *   - IndexedDB caching of compiled WebAssembly.Module (skip recompile on revisit)
 *   - ETag-based cache invalidation (only re-fetch when WASM binary changes)
 *   - Streaming compilation for first load
 *
 * Usage:
 *   import { wasmBridge } from '@/lib/wasm'
 *   await wasmBridge.init()
 *   const result = wasmBridge.compile(dslSource)
 */

import type {
  CompileResult,
  ValidateResult,
  ParseASTResult,
  DecompileResult,
  FormatResult,
  WasmBridge,
} from '@/types/dsl'

// Extend the global Window interface to include Go WASM runtime + our functions.
declare global {
  interface Window {
    Go: new () => GoInstance
    signalCompile: (dsl: string) => string
    signalValidate: (dsl: string) => string
    signalParseAST: (dsl: string) => string
    signalDecompile: (yaml: string) => string
    signalFormat: (dsl: string) => string
  }
}

interface GoInstance {
  run(instance: WebAssembly.Instance): Promise<void>
  importObject: WebAssembly.Imports
}

// Singleton state
let initPromise: Promise<void> | null = null
let isReady = false

// ─── IndexedDB WASM Module Cache ────────────────────────────

const IDB_NAME = 'signal-compiler-cache'
const IDB_STORE = 'modules'
const IDB_KEY = 'signal-compiler'

interface CachedModule {
  module: WebAssembly.Module
  etag: string | null
}

function openCacheDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1)
    req.onupgradeneeded = () => {
      req.result.createObjectStore(IDB_STORE)
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

async function getCachedModule(): Promise<CachedModule | null> {
  try {
    const db = await openCacheDB()
    return new Promise((resolve) => {
      const tx = db.transaction(IDB_STORE, 'readonly')
      const store = tx.objectStore(IDB_STORE)
      const req = store.get(IDB_KEY)
      req.onsuccess = () => resolve(req.result ?? null)
      req.onerror = () => resolve(null)
      tx.oncomplete = () => db.close()
    })
  } catch {
    return null
  }
}

async function setCachedModule(module: WebAssembly.Module, etag: string | null): Promise<void> {
  try {
    const db = await openCacheDB()
    return new Promise((resolve) => {
      const tx = db.transaction(IDB_STORE, 'readwrite')
      const store = tx.objectStore(IDB_STORE)
      store.put({ module, etag } as CachedModule, IDB_KEY)
      tx.oncomplete = () => { db.close(); resolve() }
      tx.onerror = () => { db.close(); resolve() }
    })
  } catch {
    // Cache write failure is non-fatal
  }
}

// ─── Script loader ──────────────────────────────────────────

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (typeof window.Go === 'function') {
      resolve()
      return
    }
    const script = document.createElement('script')
    script.src = src
    script.onload = () => resolve()
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`))
    document.head.appendChild(script)
  })
}

// ─── Core init ──────────────────────────────────────────────

/**
 * Initialize the Go WASM runtime and instantiate the compiler module.
 * Uses IndexedDB to cache the compiled Module — subsequent loads skip
 * the expensive compile step and only do a cheap HEAD request to check
 * if the WASM binary has changed (ETag comparison).
 */
async function init(): Promise<void> {
  if (isReady) return
  if (initPromise) return initPromise

  initPromise = (async () => {
    // 1. Load the Go JS runtime (wasm_exec.js)
    await loadScript('/wasm_exec.js')

    // 2. Instantiate Go runtime
    const go = new window.Go()

    // 3. Try to load cached compiled Module from IndexedDB
    let wasmInstance: WebAssembly.Instance | null = null
    const cached = await getCachedModule()

    if (cached) {
      // Check if WASM has changed via a cheap HEAD request
      let stale = false
      try {
        const headResp = await fetch('/signal-compiler.wasm', { method: 'HEAD' })
        const serverEtag = headResp.headers.get('etag') || headResp.headers.get('last-modified')
        stale = cached.etag !== serverEtag
      } catch {
        // Network error — use cache as fallback
      }

      if (!stale) {
        try {
          // Instantiate from cached Module — very fast (~5ms vs ~500ms+ compile)
          const result = await WebAssembly.instantiate(cached.module, go.importObject)
          wasmInstance = result
        } catch {
          // Cached module invalid — fall through to fresh load
        }
      }
    }

    if (!wasmInstance) {
      // Fresh load: fetch + compile + cache
      const resp = await fetch('/signal-compiler.wasm')
      const etag = resp.headers.get('etag') || resp.headers.get('last-modified')

      if (typeof WebAssembly.compileStreaming === 'function') {
        // Stream-compile for best first-load performance
        const module = await WebAssembly.compileStreaming(Promise.resolve(resp))
        const result = await WebAssembly.instantiate(module, go.importObject)
        wasmInstance = result
        // Cache in background
        setCachedModule(module, etag)
      } else {
        // Fallback for environments without compileStreaming
        const bytes = await resp.arrayBuffer()
        const module = await WebAssembly.compile(bytes)
        const result = await WebAssembly.instantiate(module, go.importObject)
        wasmInstance = result
        setCachedModule(module, etag)
      }
    }

    // 4. Start the Go program (runs forever via `select{}`)
    go.run(wasmInstance)

    // 5. Wait for the global functions to be registered.
    await waitForGlobals(['signalCompile', 'signalValidate', 'signalParseAST', 'signalDecompile', 'signalFormat'])

    isReady = true
  })()

  return initPromise
}

/**
 * Poll until all expected global functions are available.
 */
function waitForGlobals(names: string[], timeoutMs = 5000): Promise<void> {
  return new Promise((resolve, reject) => {
    const start = Date.now()
    const check = () => {
      const w = window as unknown as Record<string, unknown>
      const allPresent = names.every((n) => typeof w[n] === 'function')
      if (allPresent) {
        resolve()
      } else if (Date.now() - start > timeoutMs) {
        const missing = names.filter((n) => typeof w[n] !== 'function')
        reject(new Error(`WASM init timeout: missing globals [${missing.join(', ')}]`))
      } else {
        setTimeout(check, 10)
      }
    }
    check()
  })
}

/**
 * Parse a JSON string returned by WASM, with a fallback error structure.
 */
function parseResult<T>(json: string): T {
  try {
    return JSON.parse(json) as T
  } catch {
    return { error: `Failed to parse WASM response: ${json}` } as T
  }
}

function assertReady(): void {
  if (!isReady) {
    throw new Error('WASM not initialized. Call wasmBridge.init() first.')
  }
}

// ---------- Public API ----------

export const wasmBridge: WasmBridge = {
  get ready() {
    return isReady
  },

  init,

  compile(dsl: string): CompileResult {
    assertReady()
    return parseResult<CompileResult>(window.signalCompile(dsl))
  },

  validate(dsl: string): ValidateResult {
    assertReady()
    return parseResult<ValidateResult>(window.signalValidate(dsl))
  },

  parseAST(dsl: string): ParseASTResult {
    assertReady()
    return parseResult<ParseASTResult>(window.signalParseAST(dsl))
  },

  decompile(yaml: string): DecompileResult {
    assertReady()
    return parseResult<DecompileResult>(window.signalDecompile(yaml))
  },

  format(dsl: string): FormatResult {
    assertReady()
    return parseResult<FormatResult>(window.signalFormat(dsl))
  },
}
