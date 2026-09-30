interface ProviderOperationSource {
  default_base_url?: string
  supported_operations: string[]
  path_overrides?: Record<string, string>
}

interface ProtocolOperationSource {
  id: string
  method: string
  path: string
}

interface ProtocolOperationCatalog {
  id: string
  default_base_path: string
  operations: ProtocolOperationSource[]
}

export interface EffectiveProviderOperation extends ProtocolOperationSource {
  reference: string
}

const trimTrailingSlashes = (value: string): string => value.replace(/\/+$/, '')

const providerBasePath = (provider: ProviderOperationSource): string => {
  if (!provider.default_base_url) return ''
  try {
    return new URL(provider.default_base_url).pathname
  }
  catch {
    // Catalog validation owns URL correctness. Falling back to the protocol
    // path keeps the Hub usable if it receives a stale external snapshot.
    return ''
  }
}

const joinBasePath = (basePath: string, operationPath: string): string => {
  const base = trimTrailingSlashes(basePath)
  return base ? `${base}${operationPath}` : operationPath
}

const joinProtocolOperationPath = (
  basePath: string,
  defaultBasePath: string,
  operationPath: string,
): string => {
  const base = trimTrailingSlashes(basePath)
  if (!base) return operationPath

  const protocolBase = trimTrailingSlashes(defaultBasePath)
  const relativeOperationPath
    = protocolBase && protocolBase !== '/' && operationPath.startsWith(protocolBase)
      ? operationPath.slice(protocolBase.length)
      : operationPath
  return joinBasePath(base, relativeOperationPath)
}

/** Resolve the operations a provider actually exposes for one catalog protocol. */
export function providerProtocolOperations(
  provider: ProviderOperationSource,
  protocol: ProtocolOperationCatalog,
): EffectiveProviderOperation[] {
  const supported = new Set(provider.supported_operations)
  const basePath = providerBasePath(provider)

  return protocol.operations.flatMap((operation) => {
    const reference = `${protocol.id}#${operation.id}`
    if (!supported.has(reference)) return []

    const override = provider.path_overrides?.[reference]

    return [
      {
        ...operation,
        path: override
          ? joinBasePath(basePath, override)
          : joinProtocolOperationPath(basePath, protocol.default_base_path, operation.path),
        reference,
      },
    ]
  })
}
