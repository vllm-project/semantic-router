import type { RouteAlgoInput, RoutePluginInput } from '@/lib/dslMutations'
import type { ASTAlgoSpec, ASTPluginRef } from '@/types/dsl'

function astAlgoToInput(a?: ASTAlgoSpec): RouteAlgoInput | undefined {
  if (!a) return undefined
  return { algoType: a.algoType, fields: { ...a.fields } }
}

function astPluginRefToInput(p: ASTPluginRef): RoutePluginInput {
  return { name: p.name, fields: p.fields ? { ...p.fields } : undefined }
}

export { astAlgoToInput, astPluginRefToInput }
