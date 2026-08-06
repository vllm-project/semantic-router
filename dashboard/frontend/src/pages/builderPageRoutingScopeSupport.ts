import type { ASTProgram, SymbolTable } from '@/types/dsl'

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
    signalCount:
      astSignalCount || symbols?.signals?.length || sourceCount(/^\s*SIGNAL\s+/gm),
    projectionPartitionCount:
      sum((program) => program.projectionPartitions?.length ?? 0) ||
      sourceCount(/^\s*PROJECTION\s+partition\s+/gm),
    projectionScoreCount:
      sum((program) => program.projectionScores?.length ?? 0) ||
      sourceCount(/^\s*PROJECTION\s+score\s+/gm),
    projectionMappingCount:
      sum((program) => program.projectionMappings?.length ?? 0) ||
      sourceCount(/^\s*PROJECTION\s+mapping\s+/gm),
    routeCount:
      astRouteCount || symbols?.routes?.length || sourceCount(/^\s*ROUTE\s+/gm),
    pluginCount:
      astPluginCount || symbols?.plugins?.length || sourceCount(/^\s*PLUGIN\s+/gm),
    recipeCount: ast?.recipes?.length || sourceCount(/^\s*RECIPE\s+/gm),
    entrypointCount: ast?.entrypoints?.length || sourceCount(/^\s*ENTRYPOINT\s*\{/gm),
  }
}
