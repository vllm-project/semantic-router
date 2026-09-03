import type { EvaluationCampaign } from '../../types/evaluationCampaign'
import { formatMetricThreshold } from './evaluationPresentation'

export interface RequiredCheckCounts {
  passed: number
  failed: number
  unavailable: number
  total: number
}

export function requiredCheckCounts(campaign: EvaluationCampaign): RequiredCheckCounts {
  const required = campaign.decision.gates.filter((gate) => gate.disposition === 'required')
  return {
    passed: required.filter((gate) => gate.verdict === 'pass').length,
    failed: required.filter((gate) => gate.verdict === 'fail').length,
    unavailable: required.filter((gate) => gate.verdict === 'unavailable').length,
    total: required.length,
  }
}

export function formatCampaignCreatedAt(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

export function formatCampaignStatistic(value: number | undefined, signed = false): string {
  if (value === undefined) return '—'
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 4,
    signDisplay: signed ? 'exceptZero' : 'auto',
  }).format(value)
}

export function formatCampaignThreshold(threshold: {
  operator: string
  value: number
  unit?: string
}): string {
  return formatMetricThreshold(threshold)
}

export function releaseDecisionSummary(
  verdict: EvaluationCampaign['decision']['verdict'],
  counts: RequiredCheckCounts,
): string {
  switch (verdict) {
    case 'pass':
      return counts.total
        ? `Release is ready: all ${counts.total} required ${counts.total === 1 ? 'check has' : 'checks have'} passed.`
        : 'Release is ready based on the completed evaluation.'
    case 'fail':
      return counts.failed
        ? `Release is blocked because ${counts.failed} required ${counts.failed === 1 ? 'check did' : 'checks did'} not meet the release criteria.`
        : 'Release is blocked by the measured evaluation outcome.'
    case 'unavailable':
      return counts.unavailable
        ? `Release is not ready because ${counts.unavailable} required ${counts.unavailable === 1 ? 'check still needs' : 'checks still need'} results.`
        : 'Release is not ready because the required evaluation results are incomplete.'
  }
}

export function releaseNextActions(
  verdict: EvaluationCampaign['decision']['verdict'],
  counts: RequiredCheckCounts,
): string[] {
  const actions: string[] = []
  if (counts.failed) {
    actions.push(
      `Review the ${counts.failed} blocked required ${counts.failed === 1 ? 'check' : 'checks'} and address the measured regressions.`,
    )
  }
  if (counts.unavailable) {
    actions.push(
      `Complete the ${counts.unavailable} required ${counts.unavailable === 1 ? 'check' : 'checks'} that still need results.`,
    )
  }
  if (actions.length) {
    actions.push('Run the affected evaluation again before making a release decision.')
    return actions
  }
  const nextByVerdict: Record<typeof verdict, string> = {
    pass: 'Proceed with the planned release using this verified evaluation record.',
    fail: 'Review the blocked evaluation outcome and run the affected evaluation again after remediation.',
    unavailable:
      'Complete the missing evaluation results before returning to this release decision.',
  }
  return [nextByVerdict[verdict]]
}

export function gateSourceLabel(source: string): string {
  if (source === 'server_anchors') return 'Verified run records'
  if (source === 'campaign_contract') return 'Change-type policy'
  if (source === 'reference_to_fresh_live') return 'Live consistency comparison'
  if (source === 'server_attested_paired_live') return 'Controlled live comparison'
  if (source === 'server_attested_paired_live_diagnostic') return 'Controlled diagnostics'
  if (source === 'campaign_slot' || source.startsWith('campaign_slot:')) {
    return 'Bound evaluation run'
  }
  return 'Verified evaluation result'
}
