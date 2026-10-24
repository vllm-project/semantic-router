import type {
  CompileResult,
  ValidateResult,
  ParseASTResult,
  DecompileResult,
  FormatResult,
} from '@/types/dsl'
import { withRequestTimeout } from '@/utils/boundedRequest'

// The Dashboard uses the Go compiler already bundled in its backend. Authoring
// no longer waits for a multi-megabyte browser compiler download.
const pending = new Map<string, { promise: Promise<unknown>; controller: AbortController }>()

function request<T>(operation: string, source: string): Promise<T> {
  const key = `${operation}\0${source}`
  const existing = pending.get(key)
  if (existing) return existing.promise as Promise<T>
  const controller = new AbortController()
  const promise = withRequestTimeout(
    async (signal) => {
      const response = await fetch(`/api/dsl/${operation}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source }),
        signal,
      })
      if (!response.ok) throw new Error(`Compiler request failed (HTTP ${response.status}).`)
      return (await response.json()) as T
    },
    controller.signal,
    15000,
  ).finally(() => {
    if (pending.get(key)?.promise === promise) pending.delete(key)
  })
  pending.set(key, { promise, controller })
  return promise
}

export const dslCompiler = {
  cancelPending() {
    pending.forEach(({ controller }) => controller.abort())
    pending.clear()
  },
  async init(): Promise<void> {
    await request<ValidateResult>('validate', '')
  },
  compile: (source: string) => request<CompileResult>('compile', source),
  validate: (source: string) => request<ValidateResult>('validate', source),
  parseAST: (source: string) => request<ParseASTResult>('parse', source),
  decompile: (source: string) => request<DecompileResult>('decompile', source),
  format: (source: string) => request<FormatResult>('format', source),
}
