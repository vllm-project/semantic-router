import type { ASTProgram, SymbolTable } from '@/types/dsl'

export function summarizeBuilderRoutingScopes(
  ast: ASTProgram | null,
  symbols: SymbolTable | null,
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

  return {
    signalCount:
      programs.length > 0
        ? sum((program) => program.signals?.length ?? 0)
        : symbols?.signals?.length ?? 0,
    projectionPartitionCount: sum(
      (program) => program.projectionPartitions?.length ?? 0,
    ),
    projectionScoreCount: sum((program) => program.projectionScores?.length ?? 0),
    projectionMappingCount: sum((program) => program.projectionMappings?.length ?? 0),
    routeCount:
      programs.length > 0
        ? sum((program) => program.routes?.length ?? 0)
        : symbols?.routes?.length ?? 0,
    pluginCount:
      programs.length > 0
        ? sum((program) => program.plugins?.length ?? 0)
        : symbols?.plugins?.length ?? 0,
    recipeCount: ast?.recipes?.length ?? 0,
    entrypointCount: ast?.entrypoints?.length ?? 0,
  }
}
