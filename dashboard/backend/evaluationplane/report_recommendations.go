package evaluationplane

import (
	"fmt"
	"sort"
	"strings"
)

type reportRecommendation struct {
	id       string
	owner    string
	surface  string
	evidence string
	action   string
}

func (recommendation reportRecommendation) render() string {
	action := recommendation.action
	if action != "" {
		action = strings.ToUpper(action[:1]) + action[1:]
	}
	return fmt.Sprintf(
		"%s. Evidence: %s. Area: %s. Owner: %s.",
		action,
		recommendation.evidence,
		recommendation.surface,
		recommendation.owner,
	)
}

type reportMetricRecommendationRule struct {
	id         string
	metricID   string
	owner      string
	surface    string
	action     string
	comparator reportMetricRecommendationComparator
	threshold  float64
	precision  int
}

type reportMetricRecommendationComparator string

const (
	reportMetricRecommendationLessThan    reportMetricRecommendationComparator = "less_than"
	reportMetricRecommendationLessOrEqual reportMetricRecommendationComparator = "less_than_or_equal"
	reportMetricRecommendationGreaterThan reportMetricRecommendationComparator = "greater_than"
	reportMetricRecommendationAlways      reportMetricRecommendationComparator = "always"
)

func (comparator reportMetricRecommendationComparator) valid() bool {
	switch comparator {
	case reportMetricRecommendationLessThan,
		reportMetricRecommendationLessOrEqual,
		reportMetricRecommendationGreaterThan,
		reportMetricRecommendationAlways:
		return true
	default:
		return false
	}
}

func (comparator reportMetricRecommendationComparator) compare(value, threshold float64) bool {
	switch comparator {
	case reportMetricRecommendationLessThan:
		return value < threshold
	case reportMetricRecommendationLessOrEqual:
		return value <= threshold
	case reportMetricRecommendationGreaterThan:
		return value > threshold
	case reportMetricRecommendationAlways:
		return true
	default:
		return false
	}
}

var reportMetricRecommendationRules = []reportMetricRecommendationRule{
	{
		id: "AF-ROUTING-COVERAGE", metricID: "routing.coverage", owner: "Router recipe owner",
		surface:    "Routing coverage and fallback behavior",
		action:     "inspect unmatched decision traces and slice coverage before changing the model pool",
		comparator: reportMetricRecommendationLessThan, threshold: 0.95, precision: 3,
	},
	{
		id: "AF-FALLBACK", metricID: "routing.fallback_rate", owner: "Router recipe owner",
		surface:    "Routing eligibility and fallback policy",
		action:     "separate intended abstention from missing capability and verify fallback does not cross policy or trust boundaries",
		comparator: reportMetricRecommendationGreaterThan, threshold: 0.10, precision: 3,
	},
	{
		id: "AF-POOL-REDUNDANCY", metricID: "model_pool.oracle_gain", owner: "Model-pool owner",
		surface:    "Model pool composition",
		action:     "remove redundant arms or admit an arm that closes a measured capability, cost, or failure-domain gap",
		comparator: reportMetricRecommendationLessOrEqual, threshold: 0.02, precision: 3,
	},
	{
		id: "AF-POOL-COLLAPSE", metricID: "model_pool.selection_arm_coverage", owner: "Selector and model-pool owners",
		surface:    "Model eligibility, selector calibration, and utilization",
		action:     "compare arm quality and marginal contribution before deciding whether low utilization is correct dominance or selector collapse",
		comparator: reportMetricRecommendationLessThan, threshold: 0.50, precision: 3,
	},
	{
		id: "AF-UNREALIZED-POOL-VALUE", metricID: "joint.normalized_regret", owner: "Router recipe and selector owners",
		surface:    "Routing features and selector calibration",
		action:     "hold the pool fixed, inspect per-case oracle misses and decision traces, then improve feasibility recall or utility calibration",
		comparator: reportMetricRecommendationGreaterThan, threshold: 0.20, precision: 3,
	},
	{
		id: "AF-POOL-DOMINANCE", metricID: "model_pool.quality_dominated_arm_count", owner: "Model-pool owner",
		surface:    "Model pool lifecycle",
		action:     "verify the common-case dominance slice, then remove or quarantine arms that add no quality, cost, policy, modality, or failure-domain value",
		comparator: reportMetricRecommendationGreaterThan, threshold: 0, precision: 0,
	},
	{
		id: "AF-POOL-PARETO-DOMINANCE", metricID: "model_pool.pareto_dominated_arm_count", owner: "Model-pool owner",
		surface:    "Model pool quality-cost frontier",
		action:     "remove or quarantine arms dominated on the complete common-case quality-cost frontier unless they provide an explicit policy, modality, or failure-domain benefit",
		comparator: reportMetricRecommendationGreaterThan, threshold: 0, precision: 0,
	},
	{
		id: "AF-POOL-CORRELATED-FAILURE", metricID: "model_pool.mean_pairwise_failure_jaccard", owner: "Model-pool owner",
		surface:    "provider and capability failure diversity",
		action:     "admit an arm with a distinct failure domain or tighten admission for cases where the current arms fail together",
		comparator: reportMetricRecommendationGreaterThan, threshold: 0.50, precision: 3,
	},
	{
		id: "AF-POOL-WEAK-ARM", metricID: "model_pool.worst_arm_reliability", owner: "Model-pool owner",
		surface:    "Model admission and health",
		action:     "inspect per-arm failure rates and slices, then repair or quarantine the least reliable arm before pool availability masks its degradation",
		comparator: reportMetricRecommendationLessThan, threshold: 1, precision: 3,
	},
	{
		id: "AF-POOL-CAPABILITY-GAP", metricID: "model_pool.all_arm_failure_rate", owner: "Model-pool owner",
		surface:    "Model pool capability coverage",
		action:     "cluster cases where every arm fails and admit a qualified capability or reject those requests at a typed admission boundary",
		comparator: reportMetricRecommendationGreaterThan, threshold: 0, precision: 3,
	},
	{
		id: "AF-ORACLE-CAPTURE", metricID: "joint.oracle_capture_ratio", owner: "Router recipe and selector owners",
		surface:    "Routing features and selector calibration",
		action:     "hold the pool and utility contract fixed, then recover missed oracle value by slice before adding selector complexity",
		comparator: reportMetricRecommendationLessThan, threshold: 0.90, precision: 3,
	},
	{
		id: "AF-TRAJECTORY", metricID: "agentic.success_rate", owner: "Agent and Router session owners",
		surface:    "Agent session continuity, tool-loop protection, and recovery",
		action:     "evaluate step and terminal failures separately, preserve tool ownership, and test state portability under exact-step faults",
		comparator: reportMetricRecommendationLessThan, threshold: 0.90, precision: 3,
	},
	{
		id: "AF-MODALITY-CAPABILITY", metricID: "multimodal.support_rate", owner: "Router, model-pool, and serving owners",
		surface:    "Multimodal admission and model capabilities",
		action:     "separate admission, logical routing, payload transport, backend generation, and privacy failures by modality",
		comparator: reportMetricRecommendationLessThan, threshold: 1, precision: 3,
	},
	{
		id: "AF-HARD-POLICY", metricID: "safety.violation_rate", owner: "Security and recipe owners",
		surface:    "Policy enforcement and fallback boundaries",
		action:     "block promotion, identify the violating slice and enforcement path, and add a non-waivable regression case",
		comparator: reportMetricRecommendationGreaterThan, threshold: 0, precision: 3,
	},
	{
		id: "AF-SAFETY-FALSE-NEGATIVE", metricID: "safety.false_negative_rate", owner: "Security and recipe owners",
		surface:    "Unsafe-request detection and mandatory enforcement",
		action:     "block promotion, inspect unsafe cases that reached a backend, and move the invariant to static enforcement where possible",
		comparator: reportMetricRecommendationGreaterThan, threshold: 0, precision: 3,
	},
	{
		id: "AF-AGENT-PRIVACY", metricID: "agentic.privacy_exposures_per_trajectory", owner: "Agent security and Router session owners",
		surface:    "Tool arguments, session state, and model handoffs",
		action:     "block promotion and trace each exposure to its exact step, tool boundary, and model handoff before changing routing quality policy",
		comparator: reportMetricRecommendationGreaterThan, threshold: 0, precision: 3,
	},
	{
		id: "AF-CAPACITY-SATURATION", metricID: "capacity.saturation_concurrency", owner: "Serving and placement owner",
		surface:    "Queueing, batching, replicas, and accelerator placement",
		action:     "locate the SLO crossing, retry amplification, and per-arm bottleneck before changing logical routing policy",
		comparator: reportMetricRecommendationAlways, threshold: 0, precision: 0,
	},
	{
		id: "AF-ONLINE-ASSIGNMENT", metricID: "preference.propensity_coverage", owner: "Online experimentation owner",
		surface:    "Online assignment, exposure, and propensity evidence",
		action:     "do not train or claim causal preference lift until every eligible exposure records its behavior propensity and executed action",
		comparator: reportMetricRecommendationLessThan, threshold: 1, precision: 3,
	},
	{
		id: "AF-PREFERENCE-SUPPORT", metricID: "preference.effective_sample_ratio", owner: "Online experimentation owner",
		surface:    "Assignment coverage and propensity policy",
		action:     "treat the apparent lift as weakly supported, cap extreme weights, and redesign assignment coverage before updating the router online",
		comparator: reportMetricRecommendationLessThan, threshold: 0.50, precision: 3,
	},
}

func validateReportMetricRecommendationRules(rules []reportMetricRecommendationRule) error {
	if len(rules) == 0 {
		return fmt.Errorf("report metric recommendation rules are empty")
	}
	ruleIDs := make(map[string]struct{}, len(rules))
	metricIDs := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		if !portableIDPattern.MatchString(rule.id) {
			return fmt.Errorf("report metric recommendation rule id %q is invalid", rule.id)
		}
		if _, duplicate := ruleIDs[rule.id]; duplicate {
			return fmt.Errorf("report metric recommendation rule id %q is duplicated", rule.id)
		}
		ruleIDs[rule.id] = struct{}{}
		if rule.metricID == "" || rule.metricID != strings.TrimSpace(rule.metricID) {
			return fmt.Errorf("report metric recommendation rule %q has an invalid metric id", rule.id)
		}
		if _, duplicate := metricIDs[rule.metricID]; duplicate {
			return fmt.Errorf("report metric recommendation metric id %q is duplicated", rule.metricID)
		}
		metricIDs[rule.metricID] = struct{}{}
		if _, err := ResolveMetricAnalysisCatalog(rule.metricID); err != nil {
			return fmt.Errorf(
				"report metric recommendation rule %q references metric %q: %w",
				rule.id,
				rule.metricID,
				err,
			)
		}
		if !rule.comparator.valid() {
			return fmt.Errorf("report metric recommendation rule %q has an invalid comparator", rule.id)
		}
		if !finiteFloat(rule.threshold) {
			return fmt.Errorf("report metric recommendation rule %q has a non-finite threshold", rule.id)
		}
		if rule.owner == "" || rule.owner != strings.TrimSpace(rule.owner) ||
			rule.surface == "" || rule.surface != strings.TrimSpace(rule.surface) ||
			rule.action == "" || rule.action != strings.TrimSpace(rule.action) {
			return fmt.Errorf("report metric recommendation rule %q has incomplete ownership or action metadata", rule.id)
		}
	}
	return nil
}

func serverReportRecommendations(report Report, sealedLevels sealedEvidenceLevels) ([]string, error) {
	return serverReportRecommendationsWithRules(
		report,
		sealedLevels,
		reportMetricRecommendationRules,
	)
}

func serverReportRecommendationsWithRules(
	report Report,
	sealedLevels sealedEvidenceLevels,
	rules []reportMetricRecommendationRule,
) ([]string, error) {
	if err := validateReportMetricRecommendationRules(rules); err != nil {
		return nil, err
	}
	findings := make(map[string]reportRecommendation)
	metrics := make(map[string]Metric, len(report.Metrics))
	for _, metric := range report.Metrics {
		if metric.TrackID != "" && sealedLevels.ByTrack[metric.TrackID] == "E0" {
			continue
		}
		metrics[metric.ID] = metric
	}
	for _, rule := range rules {
		metric, exists := metrics[rule.metricID]
		if !exists || metric.Value == nil || !rule.comparator.compare(*metric.Value, rule.threshold) {
			continue
		}
		findings[rule.id] = reportRecommendation{
			id: rule.id, owner: rule.owner, surface: rule.surface,
			evidence: fmt.Sprintf("observed value=%.*f", rule.precision, *metric.Value),
			action:   rule.action,
		}
	}
	addModalitySliceRecommendations(metrics, findings)
	addGateRecommendations(report.Gates, findings)
	if sealedLevels.Run == "E0" {
		findings["AF-EVIDENCE-QUALIFICATION"] = reportRecommendation{
			id: "AF-EVIDENCE-QUALIFICATION", owner: "Evaluation owner",
			surface: "Evidence collection and qualification", evidence: "promotion-grade evidence is not yet available",
			action: "validate the harness, then collect qualified evidence before inferring a recipe, pool, or runtime architecture change",
		}
	}
	if len(findings) == 0 {
		return []string{"All applicable server-verified gates passed; validate on the target runtime before promotion."}, nil
	}
	ids := make([]string, 0, len(findings))
	for id := range findings {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	result := make([]string, 0, len(ids))
	for _, id := range ids {
		result = append(result, findings[id].render())
	}
	return result, nil
}

func addModalitySliceRecommendations(metrics map[string]Metric, findings map[string]reportRecommendation) {
	const prefix, suffix = "multimodal.", ".support_rate"
	for metricID, metric := range metrics {
		if metricID == "multimodal.support_rate" || !strings.HasPrefix(metricID, prefix) ||
			!strings.HasSuffix(metricID, suffix) || metric.Value == nil || *metric.Value >= 1 {
			continue
		}
		modality := strings.TrimSuffix(strings.TrimPrefix(metricID, prefix), suffix)
		if modality == "" {
			continue
		}
		id := "AF-MODALITY-" + strings.ToUpper(strings.ReplaceAll(modality, "_", "-"))
		findings[id] = reportRecommendation{
			id: id, owner: "Router, model-pool, and serving owners",
			surface:  "Multimodal admission and model capabilities",
			evidence: fmt.Sprintf("%s support=%.3f", modality, *metric.Value),
			action: fmt.Sprintf(
				"separate %s admission, routing, transport, generation, and privacy failures before changing the shared multimodal policy",
				modality,
			),
		}
	}
}

func addGateRecommendations(gates []Gate, findings map[string]reportRecommendation) {
	for _, gate := range gates {
		if gate.Verdict != "fail" && gate.Verdict != "unavailable" {
			continue
		}
		id := "AF-GATE-" + gate.ID
		findings[id] = reportRecommendation{
			id: id, owner: gate.Owner, surface: gate.Name,
			evidence: fmt.Sprintf("%s is %s", gate.Name, gate.Verdict),
			action:   fmt.Sprintf("resolve the server-verified %s evidence before promotion", gate.Name),
		}
	}
}
