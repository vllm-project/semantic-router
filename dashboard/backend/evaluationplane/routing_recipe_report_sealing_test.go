package evaluationplane

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestWorkerReportRejectsRoutingRecipeReport(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("create test run: %v", err)
	}
	draft, err := json.Marshal(workerReportFromReport(reportForRun(run, []Artifact{})))
	if err != nil {
		t.Fatalf("encode worker report: %v", err)
	}
	var forged map[string]any
	if decodeErr := json.Unmarshal(draft, &forged); decodeErr != nil {
		t.Fatalf("decode worker report fixture: %v", decodeErr)
	}
	forged["routing_recipe_report"] = map[string]any{"contract_version": RoutingRecipeEvaluationContractVersion}
	encoded, err := json.Marshal(forged)
	if err != nil {
		t.Fatalf("encode forged worker report: %v", err)
	}
	if _, err := decodeWorkerReportStrict(run.ID, encoded); err == nil || !strings.Contains(err.Error(), "unknown field \"routing_recipe_report\"") {
		t.Fatalf("forged routing recipe aggregate error=%v, want strict unknown-field rejection", err)
	}
}

func TestWorkerReportRejectsGenericRoutingRecipeMetricClaims(t *testing.T) {
	err := validateWorkerSingleRunMetricOwnership([]Metric{{
		ID: "routing_recipe.e1.eligibility_complete_rate",
	}})
	if err == nil || !strings.Contains(err.Error(), "must be read from routing_recipe_report") {
		t.Fatalf("worker generic routing recipe metric error=%v, want server-owned rejection", err)
	}
}

func TestRoutingRecipeReportWireOwnershipUsesPublishedNullAndWorkerOmission(t *testing.T) {
	report := Report{Run: Run{Mode: ModeReplay}}
	published, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("encode published report: %v", err)
	}
	var publishedEnvelope map[string]any
	if decodeErr := json.Unmarshal(published, &publishedEnvelope); decodeErr != nil {
		t.Fatalf("decode published report: %v", decodeErr)
	}
	value, present := publishedEnvelope["routing_recipe_report"]
	if !present || value != nil {
		t.Fatalf("published routing_recipe_report = %#v, present=%t; want explicit null", value, present)
	}

	worker, err := json.Marshal(workerReportFromReport(report))
	if err != nil {
		t.Fatalf("encode worker report: %v", err)
	}
	var workerEnvelope map[string]any
	if err := json.Unmarshal(worker, &workerEnvelope); err != nil {
		t.Fatalf("decode worker report: %v", err)
	}
	if _, present := workerEnvelope["routing_recipe_report"]; present {
		t.Fatal("worker envelope disclosed the server-owned routing_recipe_report field")
	}
}

func TestSealedRoutingRecipeReportReducesCompleteE1AndE2(t *testing.T) {
	manifest, records, attestation := routingRecipeReportSealFixture()
	report, err := reduceSealedRoutingRecipeReport(manifest, records, &attestation)
	if err != nil {
		t.Fatalf("reduce sealed routing recipe report: %v", err)
	}
	if report == nil || report.ContractVersion != RoutingRecipeEvaluationContractVersion ||
		report.PlanDigest != manifest.Target.Mixture.RoutingRecipePlan.PlanDigest ||
		report.E1.ExpectedDecisions != 1 || report.E1.ObservedDecisions != 1 ||
		report.E1.EligibilityComplete != 1 || report.E1.SelectedFeasible != 1 {
		t.Fatalf("sealed routing E1 report = %+v", report)
	}
	if len(report.E2.ProjectionOutcomes) != 2 || len(report.E2.TopK) != 2 ||
		!report.E2.TopK[0].Recall.Available || report.E2.TopK[0].Recall.Value != 1 ||
		!report.E2.TopK[1].Recall.Available || report.E2.TopK[1].Recall.Value != 1 ||
		!report.E2.OracleRegret.Available || report.E2.OracleRegret.Value != 0 {
		t.Fatalf("sealed routing E2 report = %+v", report.E2)
	}
	for _, projection := range report.E2.ProjectionOutcomes {
		if projection.ProjectionID == "projection:oracle-probability" {
			if !projection.Brier.Available || !projection.ECE10.Available || len(projection.Reliability) != 10 {
				t.Fatalf("sealed routing probability calibration = %+v", projection)
			}
			continue
		}
		if projection.Brier.Available || projection.Brier.Reason != "calibration_target_not_binary" ||
			projection.ECE10.Available || projection.ECE10.Reason != "calibration_target_not_binary" ||
			len(projection.Reliability) != 0 {
			t.Fatalf("sealed routing numeric projection outcome = %+v", projection)
		}
	}
	if err := validateSealedRoutingRecipeReport(report, manifest, records, &attestation); err != nil {
		t.Fatalf("exact server recomputation rejected its own report: %v", err)
	}
}

func TestSealedRoutingRecipeReportLeavesE2UnavailableWithoutCompletePool(t *testing.T) {
	manifest, records, attestation := routingRecipeReportSealFixture()
	attestation.Entries = attestation.Entries[:2] // decision plus only one eligible arm outcome
	report, err := reduceSealedRoutingRecipeReport(manifest, records, &attestation)
	if err != nil {
		t.Fatalf("reduce incomplete pool report: %v", err)
	}
	if report == nil || report.E1.ObservedDecisions != 1 || report.E1.SelectedFeasible != 1 {
		t.Fatalf("E1 must remain available when E2 pool is incomplete: %+v", report)
	}
	for _, projection := range report.E2.ProjectionOutcomes {
		for name, metric := range map[string]RoutingRecipeMetricAvailability{
			"spearman": projection.Spearman, "brier": projection.Brier, "ece": projection.ECE10,
		} {
			if metric.Available || metric.Reason != "incomplete_eligible_pool" {
				t.Fatalf("%s metric = %+v, want incomplete-pool unavailable", name, metric)
			}
		}
	}
	for _, topK := range report.E2.TopK {
		if topK.Recall.Available || topK.Recall.Reason != "incomplete_eligible_pool" {
			t.Fatalf("top-%d report = %+v", topK.K, topK)
		}
	}
	if report.E2.OracleRegret.Available || report.E2.OracleRegret.Reason != "incomplete_eligible_pool" {
		t.Fatalf("oracle regret = %+v", report.E2.OracleRegret)
	}
}

func TestSealedRoutingRecipeReportUsesOnlyEligiblePoolDenominator(t *testing.T) {
	manifest, records, attestation := routingRecipeReportSealFixture()
	decision := *attestation.Entries[0].RoutingRecipeDecision
	decision.Eligibility[1] = RoutingRecipeEligibility{
		ArmID: "arm-b", State: "ineligible", ReasonCode: "policy",
	}
	decision.RankedArmIDs = []string{"arm-a"}
	attestation.Entries[0].RoutingRecipeDecision = &decision
	attestation.Entries = attestation.Entries[:2]

	report, err := reduceSealedRoutingRecipeReport(manifest, records, &attestation)
	if err != nil {
		t.Fatalf("reduce eligible-denominator report: %v", err)
	}
	if report == nil || !report.E2.OracleRegret.Available || report.E2.OracleRegret.SampleCount != 1 {
		t.Fatalf("eligible-only E2 denominator report = %+v", report)
	}
	if len(report.E2.TopK) != 2 || !report.E2.TopK[0].Recall.Available ||
		report.E2.TopK[0].Recall.SampleCount != 1 || report.E2.TopK[1].Recall.Available ||
		report.E2.TopK[1].Recall.Reason != "ranking_shorter_than_k" {
		t.Fatalf("eligible-only top-k report = %+v", report.E2.TopK)
	}
}

func TestSealedRoutingRecipeReportRequiresServerQualityAfterDecision(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*executionAttestation)
	}{
		{
			name: "quality absent",
			mutate: func(attestation *executionAttestation) {
				attestation.Entries[2].Quality = nil
			},
		},
		{
			name: "quality observed at decision",
			mutate: func(attestation *executionAttestation) {
				observedAt := attestation.Entries[0].RoutingRecipeDecision.ObservedAt
				attestation.Entries[2].FetchedAt = &observedAt
			},
		},
		{
			name: "quality request started before decision",
			mutate: func(attestation *executionAttestation) {
				observedAt := attestation.Entries[0].RoutingRecipeDecision.ObservedAt
				completedAt := observedAt.Add(time.Second)
				attestation.Entries[2].FetchedAt = &completedAt
				attestation.Entries[2].LatencyMicroseconds = int64(2 * time.Second / time.Microsecond)
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			manifest, records, attestation := routingRecipeReportSealFixture()
			test.mutate(&attestation)
			report, err := reduceSealedRoutingRecipeReport(manifest, records, &attestation)
			if err != nil {
				t.Fatalf("reduce filtered outcome report: %v", err)
			}
			if report == nil || report.E2.OracleRegret.Available ||
				report.E2.OracleRegret.Reason != "incomplete_eligible_pool" {
				t.Fatalf("E2 accepted unbound quality/time evidence: %+v", report)
			}
		})
	}
}

func TestRoutingRecipeReportIsRequiredOnlyForLiveMixtureRouting(t *testing.T) {
	manifest, records, attestation := routingRecipeReportSealFixture()
	sealed, err := reduceSealedRoutingRecipeReport(manifest, records, &attestation)
	if err != nil {
		t.Fatalf("reduce live report: %v", err)
	}
	run := Run{
		Mode: ModeLive, TrackIDs: []TrackID{"routing"},
		Mixture: catalogMixtureFromManifest(manifest.Target.Mixture),
	}
	report := Report{Run: run, RoutingRecipeReport: sealed}
	if err := validatePublishedRoutingRecipeReportShape(report); err != nil {
		t.Fatalf("valid published live routing report rejected: %v", err)
	}
	if err := validateRoutingRecipeReportFrozenFields(run, manifest, report); err != nil {
		t.Fatalf("valid frozen live routing report rejected: %v", err)
	}

	missing := report
	missing.RoutingRecipeReport = nil
	if err := validatePublishedRoutingRecipeReportShape(missing); err == nil {
		t.Fatal("published live Mixture routing report without its server reduction was accepted")
	}

	replayManifest := manifest
	replayManifest.Mode = ModeReplay
	replayRun := run
	replayRun.Mode = ModeReplay
	replay := report
	replay.Run = replayRun
	if err := validatePublishedRoutingRecipeReportShape(replay); err == nil {
		t.Fatal("replay report containing a live routing recipe reduction was accepted")
	}
	if err := validateRoutingRecipeReportFrozenFields(replayRun, replayManifest, replay); err == nil {
		t.Fatal("frozen replay report containing a routing recipe reduction was accepted")
	}
	withoutAggregate := replay
	withoutAggregate.RoutingRecipeReport = nil
	if err := validateRoutingRecipeReportFrozenFields(replayRun, replayManifest, withoutAggregate); err != nil {
		t.Fatalf("replay report without routing recipe aggregate rejected: %v", err)
	}

	nonMixture := Report{Run: Run{Mode: ModeLive, TrackIDs: []TrackID{"routing"}}}
	if err := validatePublishedRoutingRecipeReportShape(nonMixture); err != nil {
		t.Fatalf("live non-Mixture report without routing recipe aggregate rejected: %v", err)
	}
	nonRouting := Report{Run: Run{Mode: ModeLive, TrackIDs: []TrackID{"model_pool"}, Mixture: run.Mixture}}
	if err := validatePublishedRoutingRecipeReportShape(nonRouting); err != nil {
		t.Fatalf("live non-routing Mixture report without routing recipe aggregate rejected: %v", err)
	}
}

func TestSealedRoutingRecipeReportRejectsMutatedReportPlanAndAttestation(t *testing.T) {
	manifest, records, attestation := routingRecipeReportSealFixture()
	sealed, err := reduceSealedRoutingRecipeReport(manifest, records, &attestation)
	if err != nil {
		t.Fatalf("reduce live report: %v", err)
	}

	mutatedReport := *sealed
	mutatedReport.E1.SelectedFeasible = 0
	if validationErr := validateSealedRoutingRecipeReport(&mutatedReport, manifest, records, &attestation); validationErr == nil || !strings.Contains(validationErr.Error(), "does not match the server reduction") {
		t.Fatalf("mutated report error=%v, want exact recomputation rejection", validationErr)
	}

	mutatedManifest := manifest
	mutatedMixture := *manifest.Target.Mixture
	mutatedPlan := copyRoutingRecipePlan(mutatedMixture.RoutingRecipePlan)
	mutatedPlan.Signals[0].ID = "complexity:mutated"
	mutatedPlan, err = canonicalRoutingRecipePlan(mutatedPlan)
	if err != nil {
		t.Fatalf("canonicalize mutated plan: %v", err)
	}
	mutatedMixture.RoutingRecipePlan = mutatedPlan
	mutatedManifest.Target.Mixture = &mutatedMixture
	if _, err := reduceSealedRoutingRecipeReport(mutatedManifest, records, &attestation); err == nil || !strings.Contains(err.Error(), "frozen plan") {
		t.Fatalf("mutated plan error=%v, want frozen-plan rejection", err)
	}

	mutatedAttestation := attestation
	mutatedAttestation.Entries = append([]executionAttestationEntry(nil), attestation.Entries...)
	mutatedDecision := *mutatedAttestation.Entries[0].RoutingRecipeDecision
	mutatedDecision.ObservedAt = mutatedDecision.ObservedAt.Add(time.Second)
	mutatedAttestation.Entries[0].RoutingRecipeDecision = &mutatedDecision
	if _, err := reduceSealedRoutingRecipeReport(manifest, records, &mutatedAttestation); err == nil || !strings.Contains(err.Error(), "detached") {
		t.Fatalf("mutated attestation error=%v, want broker-observation rejection", err)
	}
}

func TestReportSealPipelineOwnsOneValidatedRecordsScan(t *testing.T) {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test source path")
	}
	packageDir := filepath.Dir(currentFile)
	sealing, err := os.ReadFile(filepath.Join(packageDir, "report_sealing.go"))
	if err != nil {
		t.Fatalf("read report sealing source: %v", err)
	}
	validation, err := os.ReadFile(filepath.Join(packageDir, "report_validation.go"))
	if err != nil {
		t.Fatalf("read report validation source: %v", err)
	}
	call := "validateRecordsAndFailureSummary("
	if strings.Count(string(sealing), call) != 1 || strings.Contains(string(validation), call) {
		t.Fatal("report sealing must validate records exactly once and pass the in-memory attestation into bundle validation")
	}
}

func routingRecipeReportSealFixture() (RunManifest, recordAttestation, executionAttestation) {
	plan := testRoutingRecipePlan()
	decisionAt := time.Date(2026, 8, 31, 1, 2, 3, 0, time.UTC)
	decision := testRoutingRecipeDecision(
		"case-1", "decision-1", decisionAt, "arm-a", []string{"arm-a", "arm-b"}, 0.9, 0.8, "present",
	)
	armAAt, armBAt := decisionAt.Add(time.Second), decisionAt.Add(2*time.Second)
	armA, armB := "arm-a", "arm-b"
	qualityA, qualityB := 1.0, 0.0
	manifest := RunManifest{
		Mode: ModeLive, TrackIDs: []TrackID{"routing", "model_pool"},
		Target: ManifestTarget{Mixture: &ManifestMixture{RoutingRecipePlan: plan}},
	}
	records := recordAttestation{
		validated: true,
		PlannedCaseIDsByTrack: map[TrackID]map[string]struct{}{
			"routing": {"case-1": {}},
		},
	}
	attestation := executionAttestation{Entries: []executionAttestationEntry{
		{
			Operation: workerBrokerRouterEvaluate, TrackID: "routing", CaseID: "case-1",
			FetchedAt: &decisionAt, RoutingRecipeDecision: &decision,
		},
		{
			Operation: workerBrokerArmChatCompletion, TrackID: "model_pool", CaseID: "case-1",
			ArmID: &armA, FetchedAt: &armAAt, Quality: &qualityA,
		},
		{
			Operation: workerBrokerArmChatCompletion, TrackID: "model_pool", CaseID: "case-1",
			ArmID: &armB, FetchedAt: &armBAt, Quality: &qualityB,
		},
	}}
	return manifest, records, attestation
}
