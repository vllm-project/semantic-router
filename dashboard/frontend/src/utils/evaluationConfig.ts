import type { EvaluationDimension, EvaluationLevel } from '../types/evaluation';

export const DEFAULT_ROUTER_EVAL_ENDPOINT = 'http://localhost:8080/api/v1/eval';
export const DEFAULT_MOM_EVAL_ENDPOINT = 'http://localhost:8801';

export const LEVEL_DIMENSIONS: Record<EvaluationLevel, EvaluationDimension[]> = {
  router: ['domain', 'fact_check', 'user_feedback', 'reask'],
  mom: ['accuracy'],
};

export function getAllowedDimensionsForLevel(level: EvaluationLevel): EvaluationDimension[] {
  return LEVEL_DIMENSIONS[level];
}

export function getDefaultDimensionsForLevel(level: EvaluationLevel): EvaluationDimension[] {
  return [LEVEL_DIMENSIONS[level][0]];
}

export function normalizeDimensionsForLevel(
  level: EvaluationLevel,
  dimensions: EvaluationDimension[]
): EvaluationDimension[] {
  const allowed = new Set(getAllowedDimensionsForLevel(level));
  const normalized = dimensions.filter((dimension) => allowed.has(dimension));

  return normalized.length > 0 ? normalized : getDefaultDimensionsForLevel(level);
}

export function filterSelectedDatasetsByDimensions(
  selectedDatasets: Record<string, string[]>,
  dimensions: EvaluationDimension[]
): Record<string, string[]> {
  const allowed = new Set(dimensions);

  return Object.fromEntries(
    Object.entries(selectedDatasets).filter(([dimension]) => allowed.has(dimension as EvaluationDimension))
  );
}

export function getDefaultEndpointForLevel(level: EvaluationLevel, envoyUrl?: string): string {
  if (level === 'router') {
    return DEFAULT_ROUTER_EVAL_ENDPOINT;
  }

  return envoyUrl || DEFAULT_MOM_EVAL_ENDPOINT;
}
