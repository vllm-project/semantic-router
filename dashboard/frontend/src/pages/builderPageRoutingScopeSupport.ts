import type { ASTProgram, SymbolTable } from '@/types/dsl'

export interface BuilderRoutingScope {
  id: string
  label: string
  recipeName: string | null
  modelNames: string[]
}

const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

export function listBuilderRoutingScopes(ast: ASTProgram | null): BuilderRoutingScope[] {
  if (!ast) return []

  const entrypointsByRecipe = new Map<string, string[]>()
  for (const entrypoint of ast.entrypoints ?? []) {
    const current = entrypointsByRecipe.get(entrypoint.recipe) ?? []
    entrypointsByRecipe.set(entrypoint.recipe, [...current, ...entrypoint.modelNames])
  }

  return [
    {
      id: 'global',
      label: 'Global catalog',
      recipeName: null,
      modelNames: [],
    },
    ...(ast.recipes ?? []).map((recipe) => ({
      id: `recipe:${recipe.name}`,
      label: recipe.name,
      recipeName: recipe.name,
      modelNames: Array.from(new Set(entrypointsByRecipe.get(recipe.name) ?? [])),
    })),
  ]
}

export function resolveBuilderRoutingScope(
  ast: ASTProgram | null,
  scopeId: string,
): ASTProgram | null {
  if (!ast) return null
  if (scopeId === 'global') {
    return {
      ...ast,
      entrypoints: [],
      recipes: [],
    }
  }

  const recipeName = scopeId.startsWith('recipe:') ? scopeId.slice('recipe:'.length) : ''
  const recipe = ast.recipes?.find((candidate) => candidate.name === recipeName)
  if (!recipe) return null

  return {
    ...recipe.program,
    models: ast.models ?? [],
    entrypoints: [],
    recipes: [],
  }
}

export function chooseDefaultBuilderRoutingScope(ast: ASTProgram | null): string {
  if (!ast?.recipes?.length) return 'global'

  const hasGlobalRouting =
    (ast.signals?.length ?? 0) > 0 ||
    (ast.routes?.length ?? 0) > 0 ||
    (ast.plugins?.length ?? 0) > 0 ||
    (ast.projectionPartitions?.length ?? 0) > 0 ||
    (ast.projectionScores?.length ?? 0) > 0 ||
    (ast.projectionMappings?.length ?? 0) > 0

  return hasGlobalRouting ? 'global' : `recipe:${ast.recipes[0].name}`
}

export function mutateBuilderRecipeSource(
  source: string,
  recipeName: string | null,
  mutation: (programSource: string) => string,
): string {
  if (!recipeName) return mutation(source)

  const header = new RegExp(
    `^RECIPE\\s+${escapeRegex(recipeName)}\\s*(?:\\([^)]*\\))?\\s*\\{`,
    'm',
  ).exec(source)
  if (!header) return source

  const openBrace = source.indexOf('{', header.index)
  let depth = 0
  let closeBrace = -1
  for (let index = openBrace; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1
    if (source[index] === '}') {
      depth -= 1
      if (depth === 0) {
        closeBrace = index
        break
      }
    }
  }
  if (closeBrace < 0) return source

  const rawBody = source.slice(openBrace + 1, closeBrace)
  const bodyLines = rawBody
    .replace(/^\r?\n/, '')
    .replace(/\r?\n[ \t]*$/, '')
    .split(/\r?\n/)
  const indents = bodyLines
    .filter((line) => line.trim())
    .map((line) => line.match(/^[ \t]*/)?.[0].length ?? 0)
  const indentSize = indents.length > 0 ? Math.min(...indents) : 2
  const indent = ' '.repeat(indentSize || 2)
  const dedented = bodyLines
    .map((line) => (line.trim() ? line.slice(Math.min(indentSize, line.length)) : ''))
    .join('\n')
  const mutated = mutation(dedented).trim()
  const reindented = mutated
    ? mutated
        .split('\n')
        .map((line) => (line ? `${indent}${line}` : ''))
        .join('\n')
    : ''

  return `${source.slice(0, openBrace + 1)}\n${reindented}\n${source.slice(closeBrace)}`
}

export function summarizeBuilderRoutingScopes(
  ast: ASTProgram | null,
  symbols: SymbolTable | null,
  dslSource = '',
) {
  const programs = ast
    ? [
        ast,
        ...(ast.recipes ?? [])
          .map((recipe) => recipe.program)
          .filter((program): program is ASTProgram => Boolean(program)),
      ]
    : []
  const sum = (count: (program: ASTProgram) => number) =>
    programs.reduce((total, program) => total + count(program), 0)
  const sourceCount = (pattern: RegExp) => dslSource.match(pattern)?.length ?? 0
  const astSignalCount = sum((program) => program.signals?.length ?? 0)
  const astRouteCount = sum((program) => program.routes?.length ?? 0)
  const astPluginCount = sum((program) => program.plugins?.length ?? 0)

  return {
    signalCount: astSignalCount || symbols?.signals?.length || sourceCount(/^\s*SIGNAL\s+/gm),
    projectionPartitionCount:
      sum((program) => program.projectionPartitions?.length ?? 0) ||
      sourceCount(/^\s*PROJECTION\s+partition\s+/gm),
    projectionScoreCount:
      sum((program) => program.projectionScores?.length ?? 0) ||
      sourceCount(/^\s*PROJECTION\s+score\s+/gm),
    projectionMappingCount:
      sum((program) => program.projectionMappings?.length ?? 0) ||
      sourceCount(/^\s*PROJECTION\s+mapping\s+/gm),
    routeCount: astRouteCount || symbols?.routes?.length || sourceCount(/^\s*ROUTE\s+/gm),
    pluginCount: astPluginCount || symbols?.plugins?.length || sourceCount(/^\s*PLUGIN\s+/gm),
    recipeCount: ast?.recipes?.length || sourceCount(/^\s*RECIPE\s+/gm),
    entrypointCount: ast?.entrypoints?.length || sourceCount(/^\s*ENTRYPOINT\s*\{/gm),
  }
}
