import type { EvaluationTrackId } from '../../types/evaluationPlane'
import type { EvaluationMetric } from '../../types/evaluationReport'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'

const METRIC_LABELS: Readonly<Record<string, string>> = {
  'agentic.invalid_tool_rate': 'Invalid tool call rate',
  'agentic.mean_trajectory_steps': 'Average task steps',
  'agentic.privacy_exposures_per_trajectory': 'Privacy events per task',
  'agentic.recovery_baseline_success_rate': 'Baseline continuity',
  'agentic.recovery_cluster_count': 'Independent recovery scenarios',
  'agentic.recovery_cluster_pass_rate': 'Recovery scenario success rate',
  'agentic.recovery_cluster_pass_rate_lower_95': 'Conservative recovery success estimate',
  'agentic.recovery_distinct_seed_count': 'Distinct recovery trials',
  'agentic.recovery_max_retry_amplification': 'Additional retry load',
  'agentic.recovery_maximum_latency_ms': 'Recovery time limit',
  'agentic.recovery_mean_latency_delta_ms': 'Recovery time change',
  'agentic.recovery_minimum_cluster_count': 'Required independent recovery scenarios',
  'agentic.recovery_minimum_distinct_seed_count': 'Required recovery trial diversity',
  'agentic.recovery_pair_count': 'Matched recovery cases',
  'agentic.recovery_retry_amplification_threshold': 'Retry-load limit',
  'agentic.recovery_success_delta': 'Recovery success change',
  'agentic.recovery_treatment_success_rate': 'Recovery continuity',
  'agentic.runtime_cost_per_success': 'Cost per successful task',
  'agentic.success_rate': 'Task success rate',
  'agentic.task_attempt_count': 'Task attempts evaluated',
  'agentic.task_attempt_success_rate': 'Attempt success rate',
  'agentic.task_attempt_success_rate_lower_95': 'Attempt success lower confidence estimate',
  'agentic.task_cost_per_success_usd': 'Cost per successful attempt',
  'agentic.task_distinct_count': 'Distinct tasks evaluated',
  'agentic.task_invalid_tool_rate': 'Invalid tool call rate',
  'agentic.task_mean_score': 'Average task score',
  'agentic.task_mean_steps': 'Average task steps',
  'agentic.task_privacy_exposures_per_attempt': 'Privacy events per attempt',
  'agentic.task_pure_reasoning_attempt_count': 'Reasoning-only attempts',
  'agentic.task_reliability': 'Repeated-task reliability',
  'agentic.task_reliability_lower_95': 'Reliability lower confidence estimate',
  'agentic.task_required_tool_receipt_coverage': 'Required tool-use coverage',
  'agentic.task_score': 'Task score',
  'agentic.task_tool_required_attempt_count': 'Tool-required attempts',
  'agentic.task_total_cost_usd': 'Total task cost',
  'capacity.cost_per_successful_request': 'Cost per successful request',
  'capacity.error_rate': 'Average measurement-window error rate',
  'capacity.error_rate_cluster_range_max': 'Error-rate variation across windows',
  'capacity.error_rate_upper_bound': 'Worst-window error-rate upper estimate',
  'capacity.latency_p95_ms': '95th-percentile response time',
  'capacity.latency_p95_stability_cv_max': 'Response-time stability',
  'capacity.latency_p99_ms': '99th-percentile response time',
  'capacity.measurement_cluster_count_min': 'Independent windows per load level',
  'capacity.measurement_request_count': 'Requests measured',
  'capacity.saturation_concurrency': 'Observed saturation point',
  'capacity.saturation_concurrency_lower_bound': 'Highest tested load',
  'capacity.saturation_observed': 'Saturation observed',
  'capacity.slo_headroom': 'Capacity above the service objective',
  'capacity.success_concurrency_upper_bound': 'Highest reliable load',
  'capacity.success_rate': 'Average measurement-window success rate',
  'capacity.throughput_rps': 'Peak request throughput',
  'capacity.throughput_stability_cv_max': 'Throughput stability',
  'capacity.warmup_error_count': 'Warmup errors',
  'experiment.assignment_balance_p_value': 'Assignment balance confidence',
  'experiment.assignment_support': 'Random assignment coverage',
  'experiment.candidate_safe': 'Candidate safety result',
  'experiment.controls_operational': 'Stop and rollback readiness',
  'experiment.minimum_assignment_count': 'Required production assignments',
  'experiment.risk_budget_max_rate': 'Maximum allowed risk rate',
  'experiment.risk_event_rate': 'Observed production risk rate',
  'experiment.risk_event_upper_confidence_bound': 'Risk-rate upper confidence estimate',
  'joint.latency_p95_ms': 'End-to-end response time',
  'joint.normalized_regret': 'Normalized quality gap to the best model',
  'joint.oracle_capture_ratio': 'Share of best-available quality delivered',
  'joint.oracle_regret': 'Quality gap to the best model',
  'joint.realized_quality': 'Delivered response quality',
  'joint.reliability': 'End-to-end reliability',
  'joint.runtime_cost_per_success': 'Cost per successful response',
  'model_pool.all_arm_failure_rate': 'All-model failure rate',
  'model_pool.arm_count': 'Models evaluated',
  'model_pool.best_single_quality': 'Best single-model quality',
  'model_pool.mean_pairwise_failure_jaccard': 'Shared model failure rate',
  'model_pool.oracle_gain': 'Gain over the best single model',
  'model_pool.oracle_quality': 'Best available model quality',
  'model_pool.pareto_dominated_arm_count': 'Models dominated on quality and cost',
  'model_pool.pareto_evaluable_arm_count': 'Models with comparable quality and cost',
  'model_pool.quality_cost_shared_support_cases': 'Cases with complete quality and cost results',
  'model_pool.quality_cost_shared_support_fraction': 'Complete quality and cost coverage',
  'model_pool.quality_dominated_arm_count': 'Models dominated on quality',
  'model_pool.quality_shared_support_cases': 'Cases with complete model quality results',
  'model_pool.quality_shared_support_fraction': 'Complete model quality coverage',
  'model_pool.selection_arm_coverage': 'Model selection coverage',
  'model_pool.selection_entropy_bits': 'Model selection diversity',
  'model_pool.unique_win_rate': 'Unique model win rate',
  'model_pool.unique_wins': 'Cases with a unique best model',
  'model_pool.worst_arm_reliability': 'Least reliable model',
  'multimodal.privacy_violations': 'Multimodal privacy violations',
  'multimodal.quality': 'Multimodal response quality',
  'multimodal.support_rate': 'Supported-input success rate',
  'preference.agreement': 'Offline preference agreement',
  'preference.effective_sample_ratio': 'Usable weighted sample ratio',
  'preference.effective_sample_size': 'Usable weighted sample size',
  'preference.minimum_reward_lift': 'Required preference improvement',
  'preference.online_causal_eligible': 'Online comparison is statistically usable',
  'preference.online_effective_sample_ratio': 'Candidate usable sample ratio',
  'preference.online_effective_sample_size': 'Candidate usable sample size',
  'preference.online_ips_reward': 'Candidate preference reward estimate',
  'preference.online_minimum_effective_sample_ratio': 'Required usable sample ratio',
  'preference.online_minimum_effective_sample_size': 'Required usable sample size',
  'preference.online_outcome_coverage': 'Online outcome coverage',
  'preference.online_reward_lift': 'Preference improvement over baseline',
  'preference.online_reward_lift_passed': 'Preference improvement result',
  'preference.online_segment_count': 'Preference segments observed',
  'preference.online_segment_coverage': 'Preference segment coverage',
  'preference.online_snips_reward': 'Candidate preference reward estimate',
  'preference.propensity_coverage': 'Assignment-data coverage',
  'preference.reference_effective_sample_ratio': 'Baseline usable sample ratio',
  'preference.reference_effective_sample_size': 'Baseline usable sample size',
  'preference.reference_snips_reward': 'Baseline preference reward estimate',
  'preference.self_normalized_ips_agreement': 'Weighted offline preference agreement',
  'r2.compound_model_budget.audc': 'Quality across the tested budget',
  'r2.compound_model_budget.nauc': 'Normalized quality across budget',
  'r2.compound_model_budget.peak': 'Best observed quality',
  'r2.compound_model_budget.qnc': 'Quality at the maximum budget',
  'routing.abstention_rate': 'Routing abstention rate',
  'routing.accuracy': 'Routing accuracy',
  'routing.coverage': 'Routing decision coverage',
  'routing.fallback_rate': 'Fallback rate',
  'routing.latency_p50_ms': 'Median routing time',
  'routing.latency_p95_ms': '95th-percentile routing time',
  'routing.robustness_pass_rate': 'Workload-shift pass rate',
  'routing.robustness_worst_slice_pass_rate': 'Worst workload-slice pass rate',
  'routing.selected_arm_count': 'Models selected',
  'routing.selection_entropy_bits': 'Routing selection diversity',
  'routing.success_rate': 'Routing execution success rate',
  'routing_recipe.e1.eligibility_complete_rate': 'Complete model eligibility coverage',
  'routing_recipe.e1.selected_feasible_rate': 'Feasible selection rate',
  'routing_recipe.e2.oracle_regret': 'Quality gap to the best feasible model',
  'safety.block_accuracy': 'Blocking accuracy',
  'safety.false_negative_rate': 'Unsafe requests not blocked',
  'safety.false_positive_rate': 'Safe requests incorrectly blocked',
  'safety.hard_policy_observation_count': 'Policy checks observed',
  'safety.hard_policy_static_passed': 'Policy configuration result',
  'safety.violation_case_rate': 'Cases with a safety violation',
  'safety.violation_rate': 'Safety violations per case',
  'safety.violation_upper_95': 'Safety-violation upper confidence estimate',
}

const CAPACITY_LEVEL_LABELS: Readonly<Record<string, string>> = {
  elapsed_seconds: 'Measurement duration',
  error_rate: 'Average measurement-window error rate',
  error_rate_cluster_range: 'Error-rate variation across windows',
  error_rate_upper_bound: 'Worst-window error-rate upper estimate',
  latency_p95_cv: 'Response-time stability',
  latency_p95_ms: '95th-percentile response time',
  latency_p99_ms: '99th-percentile response time',
  measurement_cluster_count: 'Independent measurement windows',
  measurement_request_count: 'Requests measured',
  qualified: 'Service objective result',
  runtime_cost_usd: 'Runtime cost',
  success_rate: 'Request success rate',
  throughput_cv: 'Throughput stability',
  throughput_rps: 'Request throughput',
  throughput_scaling_efficiency: 'Scaling efficiency',
  warmup_error_count: 'Warmup errors',
  warmup_request_count: 'Warmup requests',
}

const MODEL_RESULT_LABELS: Readonly<Record<string, string>> = {
  marginal_contribution: 'Model contribution',
  quality: 'Model quality',
  success_rate: 'Model success rate',
}

function dynamicMetricLabel(metricID: string): string | null {
  const capacity = /^capacity\.level\.([1-9][0-9]{0,5})\.([a-z0-9_]+)$/.exec(metricID)
  if (capacity) {
    const label = CAPACITY_LEVEL_LABELS[capacity[2]]
    return label ? `${label} at ${capacity[1]} concurrent requests` : null
  }
  const model = /^model_pool\.arm\.[A-Za-z0-9_-]+\.([a-z_]+)$/.exec(metricID)
  if (model) return MODEL_RESULT_LABELS[model[1]] || null
  const modality = /^multimodal\.(audio|document|image|video)\.(quality|support_rate)$/.exec(
    metricID,
  )
  if (modality) {
    const input = `${modality[1][0].toUpperCase()}${modality[1].slice(1)}`
    return modality[2] === 'quality' ? `${input} response quality` : `${input} support rate`
  }
  if (/^routing_recipe\.e1\.(projection|signal)\.[A-Za-z0-9_-]+\.present_rate$/.test(metricID)) {
    return 'Routing input availability'
  }
  if (/^routing_recipe\.e1\.(projection|signal)\.[A-Za-z0-9_-]+\./.test(metricID)) {
    return 'Routing input reliability'
  }
  if (/^routing_recipe\.e2\.projection\.[A-Za-z0-9_-]+\.spearman$/.test(metricID)) {
    return 'Outcome ranking agreement'
  }
  if (/^routing_recipe\.e2\.projection\.[A-Za-z0-9_-]+\.(brier|ece_10)$/.test(metricID)) {
    return 'Outcome probability accuracy'
  }
  const recall = /^routing_recipe\.e2\.feasible_oracle_recall_at_([1-9][0-9]?)$/.exec(metricID)
  return recall ? `Best-model coverage in the top ${recall[1]}` : null
}

export function evaluationMetricLabel(metric: Pick<EvaluationMetric, 'id' | 'track_id'>): string {
  const label = METRIC_LABELS[metric.id] || dynamicMetricLabel(metric.id)
  if (label) return label
  if (metric.track_id)
    return `${TRACK_PRESENTATION[metric.track_id as EvaluationTrackId].label} measurement`
  return 'Evaluation measurement'
}
