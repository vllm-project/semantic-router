import type {
  ModelResearchBaseline,
  ModelResearchCampaign,
  ModelResearchGoalTemplate,
  ModelResearchRecipeSummary,
  ModelResearchStatus,
} from '../types/modelResearch'
import type { RouterModelInfo, RouterModelsInfo } from '../utils/routerRuntime'

export const GOAL_TEMPLATE_OPTIONS: Array<{ value: ModelResearchGoalTemplate; label: string; description: string }> = [
  {
    value: 'improve_accuracy',
    label: 'Improve current classifier accuracy',
    description: 'Run bounded LoRA training and offline evaluation loops against the current feedback, fact-check, jailbreak, intent, or PII classifiers.',
  },
  {
    value: 'explore_signal',
    label: 'Explore a new signal classifier',
    description: 'Start from a signal hypothesis, then train and validate a candidate classifier that still fits the current runtime contract.',
  },
]

export function formatPercent(value?: number): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'N/A'
  return `${(value * 100).toFixed(2)}%`
}

export function formatImprovementPP(value?: number): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'N/A'
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(2)}pp`
}

export function getCampaignStatusTone(status: ModelResearchStatus): 'good' | 'warn' | 'bad' | 'muted' {
  switch (status) {
    case 'completed':
      return 'good'
    case 'running':
    case 'pending':
      return 'warn'
    case 'failed':
    case 'blocked':
      return 'bad'
    default:
      return 'muted'
  }
}

export function getCampaignSubtitle(campaign: ModelResearchCampaign): string {
  const best = campaign.best_trial?.eval?.improvement_pp
  const subject =
    campaign.goal_template === 'explore_signal'
      ? campaign.signal_hypothesis || 'signal exploration'
      : campaign.target
  if (typeof best === 'number') {
    return `${subject} · best ${formatImprovementPP(best)}`
  }
  return `${subject} · ${campaign.platform}`
}

export function defaultCampaignName(goalTemplate: ModelResearchGoalTemplate, target: string): string {
  const prefix = goalTemplate === 'improve_accuracy' ? 'accuracy' : 'signal'
  return `${prefix}-${target}-${new Date().toISOString().slice(0, 10)}`
}

export function getDisplayedBaseline(
  campaign: ModelResearchCampaign | null,
  recipe: ModelResearchRecipeSummary | null,
): ModelResearchBaseline | null {
  if (campaign?.baseline) {
    return campaign.baseline
  }
  return recipe?.baseline ?? null
}

export function getCampaignDatasetLabel(campaign: ModelResearchCampaign): string {
  return campaign.overrides?.dataset_override || campaign.recipe.default_dataset
}

export function getRuntimeBaselineModel(
  runtimeModels: RouterModelsInfo | null | undefined,
  baseline: ModelResearchBaseline | null,
): RouterModelInfo | null {
  if (!runtimeModels?.models?.length || !baseline?.runtime_name) {
    return null
  }

  return (
    runtimeModels.models.find((model) => model.name.toLowerCase() === baseline.runtime_name?.toLowerCase()) ??
    null
  )
}

export function getRuntimeBaselineInventory(
  runtimeModels: RouterModelsInfo | null | undefined,
  baseline: ModelResearchBaseline | null,
): RouterModelsInfo | null {
  const matchedModel = getRuntimeBaselineModel(runtimeModels, baseline)
  if (!runtimeModels || !matchedModel) {
    return null
  }

  return {
    ...runtimeModels,
    models: [matchedModel],
    summary: {
      ...(runtimeModels.summary ?? {}),
      loaded_models: matchedModel.loaded ? 1 : 0,
      total_models: 1,
    },
  }
}

export interface ModelResearchTrendPoint {
  label: string
  round: number
  elapsedLabel: string
  elapsedMinutes: number
  accuracyPct: number
  bestAccuracyPct: number
  improvementPP: number
}

function parseTimestamp(value?: string): number | null {
  if (!value) return null
  const parsed = Date.parse(value)
  return Number.isNaN(parsed) ? null : parsed
}

export function formatDurationFromMs(durationMs: number): string {
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    return '0m'
  }

  const totalMinutes = Math.round(durationMs / 60000)
  if (totalMinutes < 1) {
    return '<1m'
  }
  if (totalMinutes < 60) {
    return `${totalMinutes}m`
  }

  const hours = Math.floor(totalMinutes / 60)
  const minutes = totalMinutes % 60
  if (minutes === 0) {
    return `${hours}h`
  }
  return `${hours}h ${minutes}m`
}

export function getCampaignElapsedLabel(campaign: ModelResearchCampaign): string {
  const start = parseTimestamp(campaign.created_at)
  const end =
    parseTimestamp(campaign.completed_at) ??
    parseTimestamp(campaign.updated_at) ??
    Date.now()

  if (start == null) {
    return 'N/A'
  }
  return formatDurationFromMs(Math.max(0, end - start))
}

export function getCompletedTrialCount(campaign: ModelResearchCampaign): number {
  return (campaign.trials ?? []).filter((trial) => !!trial.eval).length
}

export function buildTrendPoints(campaign: ModelResearchCampaign): ModelResearchTrendPoint[] {
  const points: ModelResearchTrendPoint[] = []
  const createdAt = parseTimestamp(campaign.created_at) ?? Date.now()
  const baselineAccuracyPct = (campaign.baseline_eval?.accuracy ?? 0) * 100
  let bestAccuracyPct = baselineAccuracyPct

  if (campaign.baseline_eval?.accuracy != null) {
    points.push({
      label: 'Baseline',
      round: 0,
      elapsedLabel: '0m',
      elapsedMinutes: 0,
      accuracyPct: baselineAccuracyPct,
      bestAccuracyPct,
      improvementPP: 0,
    })
  }

  const trials = [...(campaign.trials ?? [])].sort((left, right) => left.index - right.index)
  for (const trial of trials) {
    if (trial.eval?.accuracy == null) {
      continue
    }
    const trialAccuracyPct = trial.eval.accuracy * 100
    bestAccuracyPct = Math.max(bestAccuracyPct, trialAccuracyPct)
    const eventTime =
      parseTimestamp(trial.completed_at) ??
      parseTimestamp(trial.started_at) ??
      parseTimestamp(campaign.updated_at) ??
      createdAt
    const elapsedMs = Math.max(0, eventTime - createdAt)

    points.push({
      label: `Trial ${trial.index}`,
      round: trial.index,
      elapsedLabel: formatDurationFromMs(elapsedMs),
      elapsedMinutes: Math.round(elapsedMs / 60000),
      accuracyPct: trialAccuracyPct,
      bestAccuracyPct,
      improvementPP:
        trial.eval.improvement_pp ??
        (baselineAccuracyPct > 0 ? trialAccuracyPct - baselineAccuracyPct : 0),
    })
  }

  return points
}
