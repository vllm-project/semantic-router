package evaluationplane

import "fmt"

type methodMetricExpectation struct {
	Name        string
	TrackID     TrackID
	Unit        string
	Direction   string
	Value       *float64
	Interval    []float64
	SampleCount int
}

type methodMetricExpectationAdder func(
	trackID TrackID,
	id, name, unit, direction string,
	value *float64,
	count int,
	interval []float64,
)

func methodFloatPointer(value float64) *float64 {
	copyValue := value
	return &copyValue
}

func boolFloatPointer(value *bool) *float64 {
	if value == nil {
		return nil
	}
	numeric := 0.0
	if *value {
		numeric = 1
	}
	return &numeric
}

func methodMetricExpectations(methods methodRecordAttestation, selectedTracks []TrackID) map[string]methodMetricExpectation {
	expected := make(map[string]methodMetricExpectation)
	add := func(trackID TrackID, id, name, unit, direction string, value *float64, count int, interval []float64) {
		if containsTrack(selectedTracks, trackID) {
			expected[id] = methodMetricExpectation{
				Name: name, TrackID: trackID, Unit: unit, Direction: direction,
				Value: value, Interval: interval, SampleCount: count,
			}
		}
	}
	robustness := methods.Robustness
	add("routing", "routing.robustness_pass_rate", "Pinned declared-shift relation pass rate", "fraction", "higher_is_better", robustness.PassRate, robustness.PairCount, nil)
	add("routing", "routing.robustness_worst_slice_pass_rate", "Worst declared robustness-slice pass rate", "fraction", "higher_is_better", robustness.WorstSlicePassRate, robustness.PairCount, nil)

	agentTask := methods.AgentTask
	var taskAttempts, distinctTasks, toolRequiredAttempts, pureReasoningAttempts *float64
	if agentTask.AttemptCount > 0 {
		taskAttempts = methodFloatPointer(float64(agentTask.AttemptCount))
		distinctTasks = methodFloatPointer(float64(agentTask.DistinctTaskCount))
		toolRequiredAttempts = methodFloatPointer(float64(agentTask.ToolRequiredAttemptCount))
		pureReasoningAttempts = methodFloatPointer(float64(agentTask.PureReasoningAttemptCount))
	}
	add("agentic", "agentic.task_attempt_count", "Sealed agent-task attempt count", "attempts", "higher_is_better", taskAttempts, agentTask.AttemptCount, nil)
	add("agentic", "agentic.task_distinct_count", "Distinct sealed agent tasks", "tasks", "higher_is_better", distinctTasks, agentTask.AttemptCount, nil)
	add("agentic", "agentic.task_attempt_success_rate", "Agent-task attempt success rate", "fraction", "higher_is_better", agentTask.TaskSuccessRate, agentTask.AttemptCount, nil)
	add("agentic", "agentic.task_attempt_success_rate_lower_95", "One-sided 95% agent-task attempt success lower bound", "fraction", "higher_is_better", agentTask.TaskSuccessRateLower95, agentTask.AttemptCount, nil)
	add("agentic", "agentic.task_reliability", "Repeated-task all-attempt reliability", "fraction", "higher_is_better", agentTask.TaskReliability, agentTask.DistinctTaskCount, nil)
	add("agentic", "agentic.task_reliability_lower_95", "One-sided 95% repeated-task reliability lower bound", "fraction", "higher_is_better", agentTask.TaskReliabilityLower95, agentTask.DistinctTaskCount, nil)
	add("agentic", "agentic.task_mean_score", "Mean sealed agent-task score", "score", "higher_is_better", agentTask.MeanTaskScore, agentTask.AttemptCount, nil)
	add("agentic", "agentic.task_mean_steps", "Mean sealed agent-task trajectory steps", "steps", "target", agentTask.MeanTrajectorySteps, agentTask.AttemptCount, nil)
	add("agentic", "agentic.task_invalid_tool_rate", "Invalid tool-call rate in sealed agent tasks", "fraction", "lower_is_better", agentTask.InvalidToolCallRate, int(agentTask.ToolCallCount), nil)
	add("agentic", "agentic.task_tool_required_attempt_count", "Tool-required agent-task attempts", "attempts", "target", toolRequiredAttempts, agentTask.AttemptCount, nil)
	add("agentic", "agentic.task_pure_reasoning_attempt_count", "Pure-reasoning agent-task attempts", "attempts", "target", pureReasoningAttempts, agentTask.AttemptCount, nil)
	add("agentic", "agentic.task_required_tool_receipt_coverage", "Provider-executed required-tool receipt coverage", "fraction", "higher_is_better", agentTask.RequiredToolReceiptCoverage, agentTask.ToolRequiredAttemptCount, nil)
	add("agentic", "agentic.task_privacy_exposures_per_attempt", "Privacy exposures per sealed agent-task attempt", "exposures/attempt", "lower_is_better", agentTask.PrivacyExposuresPerAttempt, agentTask.AttemptCount, nil)
	add("agentic", "agentic.task_total_cost_usd", "Complete sealed agent-task cost", "USD", "lower_is_better", agentTask.TotalCostUSD, agentTask.AttemptCount, nil)
	add("agentic", "agentic.task_cost_per_success_usd", "Complete agent-task cost per successful attempt", "USD/success", "lower_is_better", agentTask.CostPerSuccessfulAttemptUSD, agentTask.SuccessfulAttemptCount, nil)

	recovery := methods.Recovery
	var recoveryPairs, recoveryClusters, minimumClusters, distinctSeeds, minimumSeeds *float64
	if recovery.PairCount > 0 {
		recoveryPairs = methodFloatPointer(float64(recovery.PairCount))
		distinctSeeds = methodFloatPointer(float64(recovery.DistinctSeedCount))
		minimumSeeds = methodFloatPointer(float64(recovery.MinimumDistinctSeedCount))
	}
	if recovery.ClusterCount > 0 {
		recoveryClusters = methodFloatPointer(float64(recovery.ClusterCount))
		minimumClusters = methodFloatPointer(float64(recovery.MinimumClusterCount))
	}
	var recoveryClusterInterval []float64
	if recovery.ClusterPassRateLower95 != nil && recovery.ClusterPassRateUpper95 != nil {
		recoveryClusterInterval = []float64{*recovery.ClusterPassRateLower95, *recovery.ClusterPassRateUpper95}
	}
	add("agentic", "agentic.recovery_pair_count", "Live fault-recovery pair count", "pairs", "higher_is_better", recoveryPairs, recovery.PairCount, nil)
	add("agentic", "agentic.recovery_cluster_count", "Independent fault-recovery cluster count", "clusters", "higher_is_better", recoveryClusters, recovery.ClusterCount, nil)
	add("agentic", "agentic.recovery_cluster_pass_rate", "All-pairs recovery pass rate across independent clusters", "fraction", "higher_is_better", recovery.ClusterPassRate, recovery.ClusterCount, recoveryClusterInterval)
	add("agentic", "agentic.recovery_cluster_pass_rate_lower_95", "One-sided 95% independent-cluster recovery lower bound", "fraction", "higher_is_better", recovery.ClusterPassRateLower95, recovery.ClusterCount, nil)
	add("agentic", "agentic.recovery_minimum_cluster_count", "Frozen minimum independent fault-recovery clusters", "clusters", "higher_is_better", minimumClusters, recovery.ClusterCount, nil)
	add("agentic", "agentic.recovery_treatment_success_rate", "All-pairs treatment success across independent clusters", "fraction", "higher_is_better", recovery.TreatmentSuccessRate, recovery.ClusterCount, nil)
	add("agentic", "agentic.recovery_baseline_success_rate", "All-pairs baseline success across independent clusters", "fraction", "higher_is_better", recovery.BaselineSuccessRate, recovery.ClusterCount, nil)
	add("agentic", "agentic.recovery_success_delta", "Cluster-weighted treatment minus baseline continuity success", "fraction", "higher_is_better", recovery.SuccessDelta, recovery.ClusterCount, nil)
	add("agentic", "agentic.recovery_mean_latency_delta_ms", "Mean cluster-worst treatment minus baseline recovery latency", "ms", "lower_is_better", recovery.MeanLatencyDeltaMS, recovery.ClusterCount, nil)
	add("agentic", "agentic.recovery_max_retry_amplification", "Maximum paired retry amplification", "ratio", "lower_is_better", recovery.MaximumRetryObserved, recovery.PairCount, nil)
	add("agentic", "agentic.recovery_maximum_latency_ms", "Frozen maximum treatment recovery latency", "ms", "lower_is_better", recovery.MaximumRecoveryLatencyMS, recovery.PairCount, nil)
	add("agentic", "agentic.recovery_retry_amplification_threshold", "Frozen maximum retry amplification", "ratio", "lower_is_better", recovery.MaximumRetryAmplification, recovery.PairCount, nil)
	add("agentic", "agentic.recovery_distinct_seed_count", "Distinct live fault-recovery seeds", "seeds", "higher_is_better", distinctSeeds, recovery.PairCount, nil)
	add("agentic", "agentic.recovery_minimum_distinct_seed_count", "Frozen minimum distinct fault-recovery seeds", "seeds", "higher_is_better", minimumSeeds, recovery.PairCount, nil)

	addProductionMetricExpectations(add, methods.Production)

	hardPolicy := methods.HardPolicy
	var observationCount *float64
	if hardPolicy.ObservationCount > 0 {
		observationCount = methodFloatPointer(float64(hardPolicy.ObservationCount))
	}
	add("safety", "safety.hard_policy_static_passed", "Runtime hard-policy static proof result", "boolean", "higher_is_better", boolFloatPointer(hardPolicy.StaticPassed), hardPolicy.ObservationCount, nil)
	add("safety", "safety.hard_policy_observation_count", "Hard-policy dynamic observation count", "observations", "higher_is_better", observationCount, hardPolicy.ObservationCount, nil)
	return expected
}

func addProductionMetricExpectations(add methodMetricExpectationAdder, production productionMethodAttestation) {
	count := production.AssignmentCount
	var minimumAssignments, minimumEffective, minimumRatio, minimumLift, segmentCount, causalEligible *float64
	if count > 0 {
		minimumAssignments = methodFloatPointer(float64(production.MinimumAssignmentCount))
		minimumEffective = methodFloatPointer(production.MinimumEffectiveSampleSize)
		minimumRatio = methodFloatPointer(production.MinimumEffectiveSampleRatio)
		minimumLift = methodFloatPointer(production.MinimumRewardLift)
		segmentCount = methodFloatPointer(float64(production.SegmentCount))
		causalEligible = methodFloatPointer(0)
		if production.CausalEligible {
			causalEligible = methodFloatPointer(1)
		}
	}
	var snipsInterval, liftInterval []float64
	if production.TargetSNIPSLower95 != nil && production.TargetSNIPSUpper95 != nil {
		snipsInterval = []float64{*production.TargetSNIPSLower95, *production.TargetSNIPSUpper95}
	}
	if production.RewardLiftLower95 != nil && production.RewardLiftUpper95 != nil {
		liftInterval = []float64{*production.RewardLiftLower95, *production.RewardLiftUpper95}
	}
	add("preference", "experiment.assignment_support", "Randomized policy-arm support", "fraction", "higher_is_better", production.AssignmentSupport, count, nil)
	add("preference", "experiment.assignment_balance_p_value", "Policy-arm assignment-balance p-value", "p-value", "higher_is_better", production.AssignmentBalancePValue, count, nil)
	add("preference", "experiment.risk_event_rate", "Production experiment risk-event rate", "fraction", "lower_is_better", production.RiskEventRate, count, nil)
	add("preference", "experiment.risk_event_upper_confidence_bound", "One-sided 95% production risk-event upper bound", "fraction", "lower_is_better", production.RiskEventUpper95, count, nil)
	add("preference", "experiment.risk_budget_max_rate", "Frozen production risk-budget maximum", "fraction", "lower_is_better", production.RiskBudgetMaxRate, count, nil)
	add("preference", "experiment.minimum_assignment_count", "Frozen minimum production assignment count", "assignments", "higher_is_better", minimumAssignments, count, nil)
	add("preference", "experiment.controls_operational", "Stop and rollback control-plane readiness", "boolean", "higher_is_better", boolFloatPointer(production.ControlsOperational), count, nil)
	add("preference", "experiment.candidate_safe", "Production candidate safety result", "boolean", "higher_is_better", boolFloatPointer(production.CandidateSafe), count, nil)
	add("preference", "preference.online_outcome_coverage", "Assignment/exposure/outcome coverage", "fraction", "higher_is_better", production.OutcomeCoverage, count, nil)
	add("preference", "preference.online_effective_sample_size", "Target-policy effective sample size", "effective samples", "higher_is_better", production.TargetEffectiveSampleSize, count, nil)
	add("preference", "preference.online_minimum_effective_sample_size", "Frozen minimum effective sample size", "effective samples", "higher_is_better", minimumEffective, count, nil)
	add("preference", "preference.online_effective_sample_ratio", "Target-policy effective-sample ratio", "fraction", "higher_is_better", production.TargetEffectiveSampleRatio, count, nil)
	add("preference", "preference.reference_effective_sample_size", "Reference-policy effective sample size", "effective samples", "higher_is_better", production.ReferenceEffectiveSampleSize, count, nil)
	add("preference", "preference.reference_effective_sample_ratio", "Reference-policy effective-sample ratio", "fraction", "higher_is_better", production.ReferenceEffectiveSampleRatio, count, nil)
	add("preference", "preference.online_minimum_effective_sample_ratio", "Frozen minimum effective-sample ratio", "fraction", "higher_is_better", minimumRatio, count, nil)
	add("preference", "preference.online_ips_reward", "Inverse-propensity target-policy reward", "fraction", "higher_is_better", production.TargetIPSReward, count, nil)
	add("preference", "preference.online_snips_reward", "Self-normalized IPS target-policy reward", "fraction", "higher_is_better", production.TargetSNIPSReward, count, snipsInterval)
	add("preference", "preference.reference_snips_reward", "Reference-policy self-normalized IPS reward", "fraction", "higher_is_better", production.ReferenceSNIPSReward, count, nil)
	add("preference", "preference.online_reward_lift", "Target-minus-reference SNIPS reward lift", "fraction", "higher_is_better", production.RewardLift, count, liftInterval)
	add("preference", "preference.minimum_reward_lift", "Frozen minimum target-vs-reference reward lift", "fraction", "higher_is_better", minimumLift, count, nil)
	add("preference", "preference.online_segment_count", "Observed preference segments", "segments", "higher_is_better", segmentCount, count, nil)
	add("preference", "preference.online_segment_coverage", "Minimum-sample segment coverage", "fraction", "higher_is_better", production.SegmentCoverage, count, nil)
	add("preference", "preference.online_causal_eligible", "Causal estimator eligibility", "boolean", "higher_is_better", causalEligible, count, nil)
	add("preference", "preference.online_reward_lift_passed", "Target-vs-reference lower-confidence-bound result", "boolean", "higher_is_better", boolFloatPointer(production.PreferencePassed), count, nil)
}

func validateServerReducedMethodMetrics(report Report, methods methodRecordAttestation) error {
	expected := methodMetricExpectations(methods, report.Run.TrackIDs)
	allIDs := methodMetricExpectations(methodRecordAttestation{
		Robustness: robustnessMethodAttestation{PairCount: 1},
		AgentTask:  agentTaskMethodAttestation{AttemptCount: 1, DistinctTaskCount: 1},
		Recovery:   recoveryMethodAttestation{PairCount: 1},
		Production: productionMethodAttestation{AssignmentCount: 1},
		HardPolicy: hardPolicyMethodAttestation{ObservationCount: 1},
	}, allTrackIDs)
	actual := make(map[string]Metric, len(report.Metrics))
	for _, metric := range report.Metrics {
		actual[metric.ID] = metric
	}
	for id := range allIDs {
		want, selected := expected[id]
		got, present := actual[id]
		if !selected {
			if present {
				return fmt.Errorf("%w: method metric %s is published for an unselected track", ErrInvalid, id)
			}
			continue
		}
		if !present || got.Name != want.Name || got.TrackID != want.TrackID || got.Unit != want.Unit || got.Direction != want.Direction ||
			got.SampleCount != want.SampleCount || (got.Value == nil) != (want.Value == nil) {
			return fmt.Errorf("%w: method metric %s does not match its server reducer", ErrInvalid, id)
		}
		if got.Value != nil && !reducedFloatsEqual(*got.Value, *want.Value) {
			return fmt.Errorf("%w: method metric %s value does not match its server reducer", ErrInvalid, id)
		}
		if !reducedIntervalsEqual(got.ConfidenceInterval, want.Interval) {
			return fmt.Errorf("%w: method metric %s interval does not match its server reducer", ErrInvalid, id)
		}
	}
	return nil
}
