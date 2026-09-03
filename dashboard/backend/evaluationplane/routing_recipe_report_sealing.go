package evaluationplane

import (
	"fmt"
	"reflect"
	"sort"
	"time"
)

// reduceSealedRoutingRecipeReport is the only publication path for a routing
// recipe report. The immutable plan, decision snapshots, post-decision quality,
// and planned case denominator all come from server-validated evidence.
func reduceSealedRoutingRecipeReport(
	manifest RunManifest,
	records recordAttestation,
	attestation *executionAttestation,
) (*RoutingRecipeEvaluationReport, error) {
	if !routingRecipeReportRequired(manifest.Mode, manifest.Target.Mixture, manifest.TrackIDs) {
		return nil, nil
	}
	if !records.validated {
		return nil, fmt.Errorf("%w: routing recipe report requires validated records", ErrInvalid)
	}
	if attestation == nil {
		return nil, fmt.Errorf("%w: live routing recipe report requires server execution attestation", ErrInvalid)
	}

	plan := manifest.Target.Mixture.RoutingRecipePlan
	expectedCaseIDs := sortedStringSet(records.PlannedCaseIDsByTrack["routing"])
	decisions := make([]RoutingRecipeDecisionSnapshot, 0, len(expectedCaseIDs))
	decisionByCase := make(map[string]RoutingRecipeDecisionSnapshot, len(expectedCaseIDs))
	for _, entry := range attestation.Entries {
		if entry.Operation != workerBrokerRouterEvaluate {
			continue
		}
		if entry.TrackID != "routing" || entry.RoutingRecipeDecision == nil || entry.FetchedAt == nil {
			return nil, fmt.Errorf("%w: routing attestation omits its server decision", ErrInvalid)
		}
		decision := *entry.RoutingRecipeDecision
		if decision.CaseID != entry.CaseID || !decision.ObservedAt.Equal(entry.FetchedAt.UTC()) {
			return nil, fmt.Errorf("%w: routing decision is detached from its broker observation", ErrInvalid)
		}
		decisions = append(decisions, decision)
		decisionByCase[decision.CaseID] = decision
	}

	outcomes := make([]RoutingRecipeOutcome, 0)
	for _, entry := range attestation.Entries {
		if entry.Operation != workerBrokerArmChatCompletion || entry.TrackID != "model_pool" ||
			entry.ArmID == nil || entry.FetchedAt == nil || entry.Quality == nil {
			continue
		}
		decision, planned := decisionByCase[entry.CaseID]
		if !planned || !routingRecipeOutcomeStartedAfterDecision(entry, decision) ||
			!routingRecipeArmIsEligible(decision, *entry.ArmID) {
			continue
		}
		outcomes = append(outcomes, RoutingRecipeOutcome{
			DecisionID: decision.DecisionID,
			CaseID:     decision.CaseID,
			ArmID:      *entry.ArmID,
			ObservedAt: entry.FetchedAt.UTC(),
			Quality:    *entry.Quality,
		})
	}

	reduced, err := ReduceRoutingRecipeEvaluation(RoutingRecipeReductionInput{
		Plan: plan, ExpectedCaseIDs: expectedCaseIDs, Decisions: decisions, Outcomes: outcomes,
	})
	if err != nil {
		return nil, fmt.Errorf("%w: reduce sealed routing recipe report: %w", ErrInvalid, err)
	}
	return &reduced, nil
}

func routingRecipeOutcomeStartedAfterDecision(
	entry executionAttestationEntry,
	decision RoutingRecipeDecisionSnapshot,
) bool {
	if entry.FetchedAt == nil || entry.LatencyMicroseconds < 0 ||
		entry.LatencyMicroseconds > int64(^uint64(0)>>1)/int64(time.Microsecond) {
		return false
	}
	startedAt := entry.FetchedAt.UTC().Add(-time.Duration(entry.LatencyMicroseconds) * time.Microsecond)
	return startedAt.After(decision.ObservedAt)
}

func validateSealedRoutingRecipeReport(
	actual *RoutingRecipeEvaluationReport,
	manifest RunManifest,
	records recordAttestation,
	attestation *executionAttestation,
) error {
	expected, err := reduceSealedRoutingRecipeReport(manifest, records, attestation)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(actual, expected) {
		return fmt.Errorf("%w: routing recipe report does not match the server reduction", ErrInvalid)
	}
	return nil
}

func routingRecipeReportRequired(mode Mode, mixture *ManifestMixture, trackIDs []TrackID) bool {
	return mode == ModeLive && mixture != nil && containsTrack(trackIDs, "routing")
}

func routingRecipeReportRequiredForRun(run Run) bool {
	return run.Mode == ModeLive && run.Mixture != nil && containsTrack(run.TrackIDs, "routing")
}

func routingRecipeArmIsEligible(decision RoutingRecipeDecisionSnapshot, armID string) bool {
	for _, eligibility := range decision.Eligibility {
		if eligibility.ArmID == armID {
			return eligibility.State == "eligible"
		}
	}
	return false
}

func sortedStringSet(values map[string]struct{}) []string {
	result := make([]string, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

// validatePublishedRoutingRecipeReportShape separates the public, server-owned
// envelope from the worker envelope. Workers never need to construct this
// aggregate; published live Mixture routing reports must always carry it.
func validatePublishedRoutingRecipeReportShape(report Report) error {
	required := routingRecipeReportRequiredForRun(report.Run)
	if required && report.RoutingRecipeReport == nil {
		return fmt.Errorf("routing_recipe_report is required for a live Mixture routing run")
	}
	if !required && report.RoutingRecipeReport != nil {
		return fmt.Errorf("routing_recipe_report is valid only for a live Mixture routing run")
	}
	if report.RoutingRecipeReport == nil {
		return nil
	}
	return validateRoutingRecipeEvaluationReportShape(
		report.Run.Mixture.RoutingRecipePlan,
		*report.RoutingRecipeReport,
	)
}

func validateRoutingRecipeReportFrozenFields(run Run, manifest RunManifest, report Report) error {
	required := routingRecipeReportRequired(manifest.Mode, manifest.Target.Mixture, manifest.TrackIDs)
	if required != routingRecipeReportRequiredForRun(run) ||
		required != routingRecipeReportRequiredForRun(report.Run) {
		return fmt.Errorf("%w: routing recipe report applicability does not match the frozen run", ErrInvalid)
	}
	if required && report.RoutingRecipeReport == nil {
		return fmt.Errorf("%w: live Mixture routing report omits its server-owned routing recipe report", ErrInvalid)
	}
	if !required {
		if report.RoutingRecipeReport != nil {
			return fmt.Errorf("%w: non-live or non-routing report contains a routing recipe report", ErrInvalid)
		}
		return nil
	}
	if manifest.Target.Mixture == nil || run.Mixture == nil || report.Run.Mixture == nil ||
		manifest.Target.Mixture.RoutingRecipePlan.PlanDigest != run.Mixture.RoutingRecipePlan.PlanDigest ||
		manifest.Target.Mixture.RoutingRecipePlan.PlanDigest != report.Run.Mixture.RoutingRecipePlan.PlanDigest {
		return fmt.Errorf("%w: routing recipe report plan differs from the frozen Mixture", ErrInvalid)
	}
	if err := validateRoutingRecipeEvaluationReportShape(
		manifest.Target.Mixture.RoutingRecipePlan,
		*report.RoutingRecipeReport,
	); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	return nil
}

func validateRoutingRecipeEvaluationReportShape(
	plan RoutingRecipePlan,
	report RoutingRecipeEvaluationReport,
) error {
	if err := ValidateRoutingRecipePlan(plan); err != nil {
		return fmt.Errorf("routing recipe report plan is invalid: %w", err)
	}
	if report.ContractVersion != RoutingRecipeEvaluationContractVersion || report.PlanDigest != plan.PlanDigest {
		return fmt.Errorf("routing recipe report does not bind the frozen plan contract")
	}
	expected := report.E1.ExpectedDecisions
	if expected < 1 || expected > routingRecipeMaxCases || report.E1.ObservedDecisions != expected ||
		report.E1.EligibilityComplete < 0 || report.E1.EligibilityComplete > expected ||
		report.E1.SelectedFeasible < 0 || report.E1.SelectedFeasible > expected {
		return fmt.Errorf("routing recipe E1 decision coverage is invalid")
	}
	if err := validateRoutingRecipeInputReports(plan.Signals, report.E1.Signals, expected); err != nil {
		return err
	}
	if err := validateRoutingRecipeInputReports(projectionSpecsAsInputs(plan.Projections), report.E1.Projections, expected); err != nil {
		return err
	}
	if report.E2.ProjectionOutcomes == nil || len(report.E2.ProjectionOutcomes) != len(plan.Projections) {
		return fmt.Errorf("routing recipe E2 projection inventory is invalid")
	}
	projectionIDs := make([]string, len(plan.Projections))
	for index, projection := range plan.Projections {
		projectionIDs[index] = projection.ID
	}
	sort.Strings(projectionIDs)
	for index, projection := range report.E2.ProjectionOutcomes {
		if projection.ProjectionID != projectionIDs[index] ||
			validateRoutingRecipeMetricAvailability(projection.Spearman, expected, -1, 1) != nil ||
			validateRoutingRecipeMetricAvailability(projection.Brier, expected, 0, 1) != nil ||
			validateRoutingRecipeMetricAvailability(projection.ECE10, expected, 0, 1) != nil {
			return fmt.Errorf("routing recipe E2 projection report is invalid")
		}
		if err := validateRoutingRecipeReliability(projection, expected); err != nil {
			return err
		}
	}
	if report.E2.TopK == nil || len(report.E2.TopK) != len(plan.TopK) {
		return fmt.Errorf("routing recipe E2 top-k inventory is invalid")
	}
	for index, topK := range report.E2.TopK {
		if topK.K != plan.TopK[index] || validateRoutingRecipeMetricAvailability(topK.Recall, expected, 0, 1) != nil {
			return fmt.Errorf("routing recipe E2 top-k report is invalid")
		}
	}
	if err := validateRoutingRecipeMetricAvailability(report.E2.OracleRegret, expected, 0, 1); err != nil {
		return fmt.Errorf("routing recipe E2 oracle regret is invalid")
	}
	return nil
}

func validateRoutingRecipeInputReports(
	specs []RoutingRecipeInputSpec,
	reports []RoutingRecipeInputAvailabilityReport,
	expected int,
) error {
	if reports == nil || len(reports) != len(specs) {
		return fmt.Errorf("routing recipe E1 input inventory is invalid")
	}
	ids := make([]string, len(specs))
	for index, spec := range specs {
		ids[index] = spec.ID
	}
	sort.Strings(ids)
	for index, input := range reports {
		if input.ID != ids[index] || input.Expected != expected || input.Present < 0 || input.Missing < 0 ||
			input.Error < 0 || input.Timeout < 0 || input.Present+input.Missing+input.Error+input.Timeout != expected {
			return fmt.Errorf("routing recipe E1 input availability is invalid")
		}
		if err := validateRoutingRecipeLatencyReport(input.Latency, input.Present); err != nil {
			return err
		}
	}
	return nil
}

func validateRoutingRecipeLatencyReport(report RoutingRecipeLatencyReport, present int) error {
	if report.SampleCount < 0 || report.SampleCount > present {
		return fmt.Errorf("routing recipe E1 latency sample count is invalid")
	}
	if report.Available {
		if report.Reason != "" || report.SampleCount < 2 || !finiteRoutingRecipeFloat(report.P50MS) ||
			!finiteRoutingRecipeFloat(report.P95MS) || report.P50MS < 0 || report.P95MS < report.P50MS {
			return fmt.Errorf("routing recipe E1 latency estimate is invalid")
		}
		return nil
	}
	if !validRoutingRecipeID(report.Reason) || report.SampleCount > 1 || report.P50MS != 0 || report.P95MS != 0 {
		return fmt.Errorf("routing recipe E1 unavailable latency is invalid")
	}
	return nil
}

func validateRoutingRecipeMetricAvailability(
	report RoutingRecipeMetricAvailability,
	expected int,
	minimum float64,
	maximum float64,
) error {
	if report.SampleCount < 0 || report.SampleCount > expected || !finiteRoutingRecipeFloat(report.Value) {
		return fmt.Errorf("routing recipe metric sample count or value is invalid")
	}
	if report.Available {
		if report.Reason != "" || report.SampleCount == 0 || report.Value < minimum || report.Value > maximum {
			return fmt.Errorf("routing recipe available metric is invalid")
		}
		return nil
	}
	if !validRoutingRecipeID(report.Reason) || report.Value != 0 {
		return fmt.Errorf("routing recipe unavailable metric is invalid")
	}
	return nil
}

func validateRoutingRecipeReliability(
	projection RoutingRecipeProjectionOutcomeReport,
	expected int,
) error {
	if !projection.Brier.Available || !projection.ECE10.Available {
		if len(projection.Reliability) != 0 {
			return fmt.Errorf("routing recipe unavailable calibration contains reliability bins")
		}
		return nil
	}
	if len(projection.Reliability) != 10 {
		return fmt.Errorf("routing recipe available calibration requires ten reliability bins")
	}
	count := 0
	for index, bin := range projection.Reliability {
		wantLower, wantUpper := float64(index)/10, float64(index+1)/10
		if bin.Lower != wantLower || bin.Upper != wantUpper || bin.Count < 0 ||
			!finiteRoutingRecipeFloat(bin.MeanPrediction) || !finiteRoutingRecipeFloat(bin.ObservedFrequency) ||
			bin.MeanPrediction < 0 || bin.MeanPrediction > 1 || bin.ObservedFrequency < 0 || bin.ObservedFrequency > 1 ||
			(bin.Count == 0 && (bin.MeanPrediction != 0 || bin.ObservedFrequency != 0)) {
			return fmt.Errorf("routing recipe reliability bin is invalid")
		}
		count += bin.Count
	}
	if count != projection.ECE10.SampleCount || count > expected {
		return fmt.Errorf("routing recipe reliability coverage is invalid")
	}
	return nil
}
