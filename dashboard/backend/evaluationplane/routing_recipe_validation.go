package evaluationplane

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var (
	routingRecipeIDPattern       = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)
	routingRecipeDigestPattern   = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	routingRecipeMetricIDPattern = regexp.MustCompile(`^routing_recipe\.(?:e1\.(?:signal|projection)\.[A-Za-z0-9_-]+\.(?:present_rate|missing_rate|error_rate|timeout_rate|latency_p50_ms|latency_p95_ms)|e1\.(?:eligibility_complete_rate|selected_feasible_rate)|e2\.projection\.[A-Za-z0-9_-]+\.(?:spearman|brier|ece_10)|e2\.feasible_oracle_recall_at_[1-9][0-9]*|e2\.oracle_regret)$`)
)

func ValidateRoutingRecipePlan(plan RoutingRecipePlan) error {
	if plan.ContractVersion != RoutingRecipePlanContractVersion || !validRoutingRecipeDigest(plan.PlanDigest) ||
		!validRoutingRecipeDigest(plan.TargetSnapshotDigest) || len(plan.ArmIDs) > routingRecipeMaxArms ||
		len(plan.Signals) > routingRecipeMaxItems || len(plan.Projections) > routingRecipeMaxItems ||
		len(plan.TopK) > routingRecipeMaxArms ||
		(len(plan.ArmIDs) == 0 && len(plan.TopK) != 0) ||
		(len(plan.ArmIDs) > 0 && len(plan.TopK) == 0) {
		return fmt.Errorf("routing recipe plan is invalid")
	}
	canonical, err := canonicalRoutingRecipePlan(plan)
	if err != nil || canonical.PlanDigest != plan.PlanDigest {
		return fmt.Errorf("routing recipe plan digest does not bind its canonical body")
	}
	arms := make(map[string]struct{}, len(plan.ArmIDs))
	for _, armID := range plan.ArmIDs {
		if !validRoutingRecipeID(armID) {
			return fmt.Errorf("routing recipe arm id is invalid")
		}
		if _, duplicate := arms[armID]; duplicate {
			return fmt.Errorf("routing recipe arms must be unique")
		}
		arms[armID] = struct{}{}
	}
	if plan.FallbackArmID != "" {
		if _, present := arms[plan.FallbackArmID]; !present {
			return fmt.Errorf("routing recipe fallback arm is outside the frozen pool")
		}
	}
	if err := validateRoutingRecipeInputSpecs(plan.Signals); err != nil {
		return err
	}
	if err := validateRoutingRecipeProjectionSpecs(plan.Projections); err != nil {
		return err
	}
	previous := 0
	for _, k := range plan.TopK {
		if k <= previous || k > len(plan.ArmIDs) {
			return fmt.Errorf("routing recipe top-k values must be strictly increasing frozen arm counts")
		}
		previous = k
	}
	return nil
}

func ValidateRoutingRecipeDecisionSnapshot(plan RoutingRecipePlan, snapshot RoutingRecipeDecisionSnapshot) error {
	if err := ValidateRoutingRecipePlan(plan); err != nil {
		return err
	}
	if err := validateRoutingRecipeDecisionSnapshotShape(snapshot); err != nil {
		return err
	}
	if snapshot.PlanDigest != plan.PlanDigest ||
		len(snapshot.Signals) != len(plan.Signals) || len(snapshot.Projections) != len(plan.Projections) ||
		len(snapshot.Eligibility) != len(plan.ArmIDs) || len(snapshot.RankedArmIDs) > len(plan.ArmIDs) {
		return fmt.Errorf("routing decision snapshot does not match its frozen plan")
	}
	if err := validateRoutingRecipeObservedInputs(plan.Signals, snapshot.Signals, false); err != nil {
		return err
	}
	if err := validateRoutingRecipeObservedInputs(projectionSpecsAsInputs(plan.Projections), snapshot.Projections, true); err != nil {
		return err
	}
	eligibilityState := make(map[string]string, len(plan.ArmIDs))
	seenEligibility := make(map[string]struct{}, len(plan.ArmIDs))
	for _, item := range snapshot.Eligibility {
		if _, allowed := frozenRoutingRecipeArmSet(plan)[item.ArmID]; !allowed {
			return fmt.Errorf("routing decision eligibility names an arm outside the frozen pool")
		}
		if _, duplicate := seenEligibility[item.ArmID]; duplicate || !validRoutingRecipeEligibility(item) {
			return fmt.Errorf("routing decision eligibility is invalid")
		}
		seenEligibility[item.ArmID] = struct{}{}
		eligibilityState[item.ArmID] = item.State
	}
	ranked := make(map[string]struct{}, len(snapshot.RankedArmIDs))
	for _, armID := range snapshot.RankedArmIDs {
		state, frozen := eligibilityState[armID]
		selectedWithoutRecommendations := state == "unavailable" &&
			armID == snapshot.SelectedArmID && len(snapshot.RankedArmIDs) == 1
		if !frozen || (state != "eligible" && !selectedWithoutRecommendations) {
			return fmt.Errorf("routing decision ranking includes a non-eligible arm")
		}
		if _, duplicate := ranked[armID]; duplicate {
			return fmt.Errorf("routing decision ranking contains duplicate arms")
		}
		ranked[armID] = struct{}{}
	}
	switch snapshot.SelectionStatus {
	case "selected":
		if err := validateFinalRoutingRecipeSelection(snapshot, eligibilityState); err != nil {
			return fmt.Errorf("selected routing decision: %w", err)
		}
	case "fallback":
		if plan.FallbackArmID == "" || snapshot.SelectedArmID != plan.FallbackArmID {
			return fmt.Errorf("fallback routing decision must select its frozen fallback arm")
		}
		if err := validateFinalRoutingRecipeSelection(snapshot, eligibilityState); err != nil {
			return fmt.Errorf("fallback routing decision: %w", err)
		}
	case "abstained", "error", "unavailable":
		if snapshot.SelectedArmID != "" {
			return fmt.Errorf("non-final routing decision cannot claim a selected arm")
		}
	default:
		return fmt.Errorf("routing decision selection status is invalid")
	}
	return nil
}

func validateFinalRoutingRecipeSelection(
	snapshot RoutingRecipeDecisionSnapshot,
	eligibilityState map[string]string,
) error {
	state, frozen := eligibilityState[snapshot.SelectedArmID]
	if snapshot.SelectedArmID == "" || !frozen {
		return fmt.Errorf("must bind a selected frozen arm")
	}
	switch state {
	case "eligible":
		if len(snapshot.RankedArmIDs) == 0 || snapshot.RankedArmIDs[0] != snapshot.SelectedArmID {
			return fmt.Errorf("eligible selection must be the first recommended arm")
		}
	case "ineligible":
		// Preserve the selector outcome even when it violates the recommendation
		// set; selected_feasible_rate must be able to observe this condition.
	case "unavailable":
		if len(snapshot.RankedArmIDs) != 1 || snapshot.RankedArmIDs[0] != snapshot.SelectedArmID {
			return fmt.Errorf("selection without recommendation evidence must be its only ranked arm")
		}
	default:
		return fmt.Errorf("cannot claim selection from failed eligibility evidence")
	}
	return nil
}

// validateRoutingRecipeDecisionSnapshotShape is the manifest-independent
// persistence boundary. Exact plan membership and value kinds are validated by
// ValidateRoutingRecipeDecisionSnapshot once the immutable manifest is loaded.
func validateRoutingRecipeDecisionSnapshotShape(snapshot RoutingRecipeDecisionSnapshot) error {
	if snapshot.ContractVersion != RoutingDecisionEvidenceContractVersion ||
		!validRoutingRecipeID(snapshot.DecisionID) || !validRoutingRecipeDigest(snapshot.PlanDigest) ||
		!validRoutingRecipeID(snapshot.CaseID) || snapshot.ObservedAt.IsZero() ||
		len(snapshot.Signals) > routingRecipeMaxItems || len(snapshot.Projections) > routingRecipeMaxItems ||
		len(snapshot.Eligibility) > routingRecipeMaxArms || len(snapshot.RankedArmIDs) > routingRecipeMaxArms {
		return fmt.Errorf("routing decision snapshot shape is invalid")
	}
	if err := validateRoutingRecipeObservedInputShape(snapshot.Signals, false); err != nil {
		return err
	}
	if err := validateRoutingRecipeObservedInputShape(snapshot.Projections, true); err != nil {
		return err
	}
	seenEligibility := make(map[string]struct{}, len(snapshot.Eligibility))
	for _, item := range snapshot.Eligibility {
		if _, duplicate := seenEligibility[item.ArmID]; duplicate || !validRoutingRecipeEligibility(item) {
			return fmt.Errorf("routing decision eligibility shape is invalid")
		}
		seenEligibility[item.ArmID] = struct{}{}
	}
	seenRanked := make(map[string]struct{}, len(snapshot.RankedArmIDs))
	for _, armID := range snapshot.RankedArmIDs {
		if !validRoutingRecipeID(armID) {
			return fmt.Errorf("routing decision ranking shape is invalid")
		}
		if _, duplicate := seenRanked[armID]; duplicate {
			return fmt.Errorf("routing decision ranking shape is invalid")
		}
		seenRanked[armID] = struct{}{}
	}
	switch snapshot.SelectionStatus {
	case "selected", "fallback":
		if !validRoutingRecipeID(snapshot.SelectedArmID) {
			return fmt.Errorf("final routing decision shape is invalid")
		}
	case "abstained", "error", "unavailable":
		if snapshot.SelectedArmID != "" {
			return fmt.Errorf("non-final routing decision shape is invalid")
		}
	default:
		return fmt.Errorf("routing decision selection status is invalid")
	}
	return nil
}

func validateRoutingRecipeObservedInputShape(observed []RoutingRecipeObservedInput, projection bool) error {
	seen := make(map[string]struct{}, len(observed))
	for _, item := range observed {
		if !validRoutingRecipeInputID(item.ID, projection) ||
			(item.LatencyMS != nil && (!finiteRoutingRecipeFloat(*item.LatencyMS) || *item.LatencyMS < 0)) {
			return fmt.Errorf("routing decision input shape is invalid")
		}
		if _, duplicate := seen[item.ID]; duplicate {
			return fmt.Errorf("routing decision input shape is invalid")
		}
		seen[item.ID] = struct{}{}
		switch item.State {
		case "present":
			if item.ErrorCode != "" || (item.Value != nil && !finiteRoutingRecipeFloat(*item.Value)) {
				return fmt.Errorf("present routing decision input shape is invalid")
			}
		case "missing", "timeout":
			if item.Value != nil || item.ErrorCode != "" {
				return fmt.Errorf("unavailable routing decision input shape is invalid")
			}
		case "error":
			if item.Value != nil || !validRoutingRecipeID(item.ErrorCode) {
				return fmt.Errorf("failed routing decision input shape is invalid")
			}
		default:
			return fmt.Errorf("routing decision input state is invalid")
		}
	}
	return nil
}

func validateRoutingRecipeReductionInput(input RoutingRecipeReductionInput) (map[string]RoutingRecipeDecisionSnapshot, error) {
	if err := ValidateRoutingRecipePlan(input.Plan); err != nil {
		return nil, err
	}
	if len(input.ExpectedCaseIDs) == 0 || len(input.ExpectedCaseIDs) > routingRecipeMaxCases || len(input.Decisions) != len(input.ExpectedCaseIDs) || len(input.Outcomes) > routingRecipeMaxOutcomes {
		return nil, fmt.Errorf("routing recipe expected decision matrix is invalid")
	}
	expected := make(map[string]struct{}, len(input.ExpectedCaseIDs))
	for _, caseID := range input.ExpectedCaseIDs {
		if !validRoutingRecipeID(caseID) {
			return nil, fmt.Errorf("routing recipe expected case id is invalid")
		}
		if _, duplicate := expected[caseID]; duplicate {
			return nil, fmt.Errorf("routing recipe expected cases must be unique")
		}
		expected[caseID] = struct{}{}
	}
	byCase := make(map[string]RoutingRecipeDecisionSnapshot, len(input.Decisions))
	seenDecisionIDs := make(map[string]struct{}, len(input.Decisions))
	for _, snapshot := range input.Decisions {
		if _, planned := expected[snapshot.CaseID]; !planned {
			return nil, fmt.Errorf("routing decision is outside the expected case set")
		}
		if _, duplicate := byCase[snapshot.CaseID]; duplicate {
			return nil, fmt.Errorf("routing decision case is duplicated")
		}
		if _, duplicate := seenDecisionIDs[snapshot.DecisionID]; duplicate {
			return nil, fmt.Errorf("routing decision id is duplicated")
		}
		if err := ValidateRoutingRecipeDecisionSnapshot(input.Plan, snapshot); err != nil {
			return nil, err
		}
		byCase[snapshot.CaseID] = snapshot
		seenDecisionIDs[snapshot.DecisionID] = struct{}{}
	}
	if len(byCase) != len(expected) {
		return nil, fmt.Errorf("routing decision coverage is incomplete")
	}
	return byCase, nil
}

func validateRoutingRecipeInputSpecs(specs []RoutingRecipeInputSpec) error {
	seen := make(map[string]struct{}, len(specs))
	for _, spec := range specs {
		if !validRoutingRecipeInputID(spec.ID, false) || (spec.ValueKind != "numeric" && spec.ValueKind != "none") {
			return fmt.Errorf("routing recipe input specification is invalid")
		}
		if _, duplicate := seen[spec.ID]; duplicate {
			return fmt.Errorf("routing recipe input specifications must be unique")
		}
		seen[spec.ID] = struct{}{}
	}
	return nil
}

func validateRoutingRecipeProjectionSpecs(specs []RoutingRecipeProjectionSpec) error {
	seen := make(map[string]struct{}, len(specs))
	for _, spec := range specs {
		if !validRoutingRecipeInputID(spec.ID, true) || (spec.ValueKind != "numeric" && spec.ValueKind != "probability") ||
			(spec.OutcomeBinding != "selected_pool_quality" && spec.OutcomeBinding != "selected_is_oracle") {
			return fmt.Errorf("routing recipe projection specification is invalid")
		}
		if _, duplicate := seen[spec.ID]; duplicate {
			return fmt.Errorf("routing recipe projection specifications must be unique")
		}
		seen[spec.ID] = struct{}{}
	}
	return nil
}

func projectionSpecsAsInputs(specs []RoutingRecipeProjectionSpec) []RoutingRecipeInputSpec {
	inputs := make([]RoutingRecipeInputSpec, len(specs))
	for index, spec := range specs {
		inputs[index] = RoutingRecipeInputSpec{ID: spec.ID, ValueKind: spec.ValueKind}
	}
	return inputs
}

func validateRoutingRecipeObservedInputs(specs []RoutingRecipeInputSpec, observed []RoutingRecipeObservedInput, projection bool) error {
	expected := make(map[string]RoutingRecipeInputSpec, len(specs))
	for _, spec := range specs {
		expected[spec.ID] = spec
	}
	seen := make(map[string]struct{}, len(observed))
	for _, item := range observed {
		spec, declared := expected[item.ID]
		if !declared || !validRoutingRecipeInputID(item.ID, projection) || (item.LatencyMS != nil && (!finiteRoutingRecipeFloat(*item.LatencyMS) || *item.LatencyMS < 0)) {
			return fmt.Errorf("routing decision input is outside the frozen plan")
		}
		if _, duplicate := seen[item.ID]; duplicate {
			return fmt.Errorf("routing decision input is duplicated")
		}
		seen[item.ID] = struct{}{}
		if item.State == "present" {
			if item.ErrorCode != "" || (spec.ValueKind == "none" && item.Value != nil) || (spec.ValueKind != "none" && item.Value == nil) ||
				(item.Value != nil && (!finiteRoutingRecipeFloat(*item.Value) || (spec.ValueKind == "probability" && (*item.Value < 0 || *item.Value > 1)))) {
				return fmt.Errorf("present routing decision input has an invalid value")
			}
			continue
		}
		if item.State != "missing" && item.State != "error" && item.State != "timeout" {
			return fmt.Errorf("routing decision input state is invalid")
		}
		if item.Value != nil || (item.State == "error" && !validRoutingRecipeID(item.ErrorCode)) || (item.State != "error" && item.ErrorCode != "") {
			return fmt.Errorf("non-present routing decision input is invalid")
		}
	}
	if len(seen) != len(expected) {
		return fmt.Errorf("routing decision input coverage is incomplete")
	}
	return nil
}

func validRoutingRecipeEligibility(item RoutingRecipeEligibility) bool {
	if !validRoutingRecipeID(item.ArmID) {
		return false
	}
	if item.State == "eligible" {
		return item.ReasonCode == "none"
	}
	if item.State != "ineligible" && item.State != "error" && item.State != "timeout" && item.State != "unavailable" {
		return false
	}
	return item.ReasonCode != "none" && validRoutingRecipeID(item.ReasonCode)
}

func frozenRoutingRecipeArmSet(plan RoutingRecipePlan) map[string]struct{} {
	result := make(map[string]struct{}, len(plan.ArmIDs))
	for _, armID := range plan.ArmIDs {
		result[armID] = struct{}{}
	}
	return result
}

func validRoutingRecipeID(value string) bool {
	return routingRecipeIDPattern.MatchString(value)
}

// validRoutingRecipeInputID validates runtime signal keys independently from
// generic case, decision, arm, reason, and error IDs. Runtime keys contain one
// or two colon separators; every segment and the complete key remain bounded
// by the metric subject encoding contract.
func validRoutingRecipeInputID(value string, projection bool) bool {
	if value == "" || value != strings.TrimSpace(value) || len(value) > 128 {
		return false
	}
	parts := strings.Split(value, ":")
	if len(parts) != 2 && len(parts) != 3 {
		return false
	}
	for _, part := range parts {
		if !routingRecipeIDPattern.MatchString(part) {
			return false
		}
	}
	signalType := parts[0]
	if signalType != strings.ToLower(signalType) {
		return false
	}
	if projection {
		return signalType == routerconfig.SignalTypeProjection && len(parts) == 2
	}
	if signalType == routerconfig.SignalTypeProjection {
		return false
	}
	if signalType == routerconfig.ProjectionInputKBMetric {
		return len(parts) == 3
	}
	if !routerconfig.IsSupportedSignalType(signalType) {
		return false
	}
	return len(parts) == 2 || signalType == routerconfig.SignalTypeClassifier
}

func validRoutingRecipeDigest(value string) bool {
	return strings.HasPrefix(value, "sha256:") && len(value) == len("sha256:")+64 && routingRecipeDigestPattern.MatchString(value)
}

// isRoutingRecipeMetricID identifies the reserved server-owned metric
// namespace. Routing-recipe results are published only through the structured
// server reduction, never as worker-authored generic metrics.
func isRoutingRecipeMetricID(value string) bool {
	return routingRecipeMetricIDPattern.MatchString(value)
}

func finiteRoutingRecipeFloat(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// canonicalRoutingRecipePlan derives the only permitted plan digest from its
// semantic body. It is intentionally independent of the caller's supplied
// digest and normalizes all declared set-like dimensions before hashing.
func canonicalRoutingRecipePlan(draft RoutingRecipePlan) (RoutingRecipePlan, error) {
	plan := draft
	plan.PlanDigest = ""
	plan.ArmIDs = append([]string{}, draft.ArmIDs...)
	plan.Signals = append([]RoutingRecipeInputSpec{}, draft.Signals...)
	plan.Projections = append([]RoutingRecipeProjectionSpec{}, draft.Projections...)
	plan.TopK = append([]int{}, draft.TopK...)
	sort.Strings(plan.ArmIDs)
	sort.Slice(plan.Signals, func(left, right int) bool { return plan.Signals[left].ID < plan.Signals[right].ID })
	sort.Slice(plan.Projections, func(left, right int) bool { return plan.Projections[left].ID < plan.Projections[right].ID })
	sort.Ints(plan.TopK)
	digest, err := canonicalValueDigest(struct {
		ContractVersion, TargetSnapshotDigest string
		ArmIDs                                []string
		FallbackArmID                         string
		Signals                               []RoutingRecipeInputSpec
		Projections                           []RoutingRecipeProjectionSpec
		TopK                                  []int
	}{plan.ContractVersion, plan.TargetSnapshotDigest, plan.ArmIDs, plan.FallbackArmID, plan.Signals, plan.Projections, plan.TopK})
	if err != nil {
		return RoutingRecipePlan{}, fmt.Errorf("digest routing recipe plan body: %w", err)
	}
	plan.PlanDigest = digest
	return plan, nil
}

func sortedRoutingRecipeIDs(values []string) []string {
	result := append([]string(nil), values...)
	sort.Strings(result)
	return result
}
