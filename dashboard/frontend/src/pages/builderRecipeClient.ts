import { wasmBridge } from '@/lib/wasm'
import type { CompileResult } from '@/types/dsl'
import type { RoutingRecipe } from '@/utils/routingManagementApi'

export interface BuilderRecipeTarget {
  id: string
  name: string
  description?: string
  recipeRevision: number
}

export interface CompiledBuilderRecipe {
  document: Record<string, unknown>
  source: string
  preview: string
}

const recipeEnvelope = (target: BuilderRecipeTarget, document: Record<string, unknown>): string =>
  JSON.stringify(
    {
      version: 'v0.4',
      recipes: [
        {
          id: target.id,
          revision: target.recipeRevision,
          name: target.name,
          ...(target.description ? { description: target.description } : {}),
          document,
        },
      ],
    },
    null,
    2,
  )

const compileFailure = (result: CompileResult): string | null => {
  if (result.error?.trim()) return result.error.trim()
  const errors = result.diagnostics?.filter((diagnostic) => diagnostic.level === 'error') ?? []
  return errors.length > 0 ? errors.map((diagnostic) => diagnostic.message).join('\n') : null
}

export function loadManagedRecipeSource(recipe: RoutingRecipe): CompiledBuilderRecipe {
  const sourceResult = wasmBridge.decompile(recipeEnvelope(recipe, recipe.document))
  if (sourceResult.error || !sourceResult.dsl.trim()) {
    throw new Error(sourceResult.error || 'The selected Recipe could not be opened in Builder.')
  }
  return compileBuilderRecipe(sourceResult.dsl, recipe)
}

/**
 * Compile exactly one Recipe and return only the model-free Management document.
 * The compiler owns projection, so provider, Model, and Entrypoint state can never
 * cross the Builder/Management boundary.
 */
export function compileBuilderRecipe(
  source: string,
  target: BuilderRecipeTarget,
): CompiledBuilderRecipe {
  const result = wasmBridge.compile(source)
  const failure = compileFailure(result)
  if (failure) throw new Error(failure)

  const documents = result.recipeDocuments ?? []
  if (documents.length !== 1) {
    throw new Error(
      'Builder requires exactly one Recipe. Models and Entrypoints belong in Routing.',
    )
  }

  const compiled = documents[0]
  const projectedSource = wasmBridge.decompile(recipeEnvelope(target, compiled.document))
  if (projectedSource.error || !projectedSource.dsl.trim()) {
    throw new Error(projectedSource.error || 'The Recipe draft could not be projected.')
  }

  return {
    document: compiled.document,
    source: projectedSource.dsl,
    preview: JSON.stringify(compiled.document, null, 2),
  }
}

export function projectImportedRecipe(
  input: string,
  target: BuilderRecipeTarget,
): CompiledBuilderRecipe {
  const decompiled = wasmBridge.decompile(input)
  if (decompiled.error || !decompiled.dsl.trim()) {
    throw new Error(decompiled.error || 'The imported Recipe could not be read.')
  }
  return compileBuilderRecipe(decompiled.dsl, target)
}
