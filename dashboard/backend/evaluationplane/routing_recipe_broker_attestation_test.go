package evaluationplane

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"
)

func TestRoutingRecipeBrokerNormalizesExactInputsEligibilityAndRanking(t *testing.T) {
	manifest := routingRecipeBrokerTestManifest(t)
	observedAt := time.Date(2026, 8, 31, 10, 11, 12, 13, time.UTC)
	selectedModel, selectedArmID, selectionStatus := "model-fast", "arm-fast", "selected"
	entry := executionAttestationEntry{
		RequestID: 7, Operation: workerBrokerRouterEvaluate, TrackID: "routing",
		CaseID: "case-1", AttemptID: "attempt-1", UpstreamAttempted: true, Success: true,
		SelectedModel: &selectedModel, ArmID: &selectedArmID, SelectionStatus: &selectionStatus,
	}
	response := workerBrokerResponse{
		Success: true, FetchedAt: observedAt,
		Payload: map[string]any{
			"signal_confidences": map[string]any{
				"domain:reasoning":              json.Number("0.91"),
				"classifier:risk:RISKY":         json.Number("0.74"),
				"projection:oracle-probability": json.Number("0.83"),
			},
			"signal_values": map[string]any{
				"domain:reasoning":         json.Number("0.52"),
				"structure:many_questions": json.Number("4"),
			},
			"signal_errors": map[string]any{
				"classifier:risk:RISKY": "classifier_evaluation_timeout",
				"language:english":      "classifier_unavailable",
			},
			"decision_result": map[string]any{
				"matched_signals": map[string]any{"domains": []any{"reasoning"}},
				"unmatched_signals": map[string]any{
					"context": []any{"turns"}, "metadata": []any{"other"},
				},
			},
			"recommended_models": []any{"model-fast", "model-strong"},
		},
	}
	snapshot := routingRecipeDecisionFromBrokerResponse(
		manifest,
		workerBrokerRequest{ID: 7, Operation: workerBrokerRouterEvaluate, TrackID: "routing", CaseID: "case-1"},
		response,
		entry,
	)
	if snapshot == nil {
		t.Fatal("live routing response omitted its broker-owned decision snapshot")
	}
	if snapshot.DecisionID != "routing-decision-7" || snapshot.PlanDigest != manifest.Target.Mixture.RoutingRecipePlan.PlanDigest ||
		!snapshot.ObservedAt.Equal(observedAt) || snapshot.SelectionStatus != "selected" ||
		snapshot.SelectedArmID != "arm-fast" || len(snapshot.RankedArmIDs) != 2 ||
		snapshot.RankedArmIDs[0] != "arm-fast" || snapshot.RankedArmIDs[1] != "arm-strong" {
		t.Fatalf("normalized routing identity/selection = %+v", snapshot)
	}
	assertRoutingRecipeObservedInput(t, snapshot.Signals, "domain:reasoning", "present", 0.52, "")
	assertRoutingRecipeObservedInput(t, snapshot.Signals, "structure:many_questions", "present", 4, "")
	assertRoutingRecipeObservedInput(t, snapshot.Signals, "classifier:risk:RISKY", "timeout", 0, "")
	assertRoutingRecipeObservedInput(t, snapshot.Signals, "language:english", "error", 0, "classifier_unavailable")
	assertRoutingRecipeObservedInput(t, snapshot.Signals, "context:turns", "present", 0, "")
	assertRoutingRecipeObservedInput(t, snapshot.Signals, "metadata:absent", "missing", 0, "")
	assertRoutingRecipeObservedInput(t, snapshot.Projections, "projection:oracle-probability", "present", 0.83, "")
	for _, input := range append(append([]RoutingRecipeObservedInput{}, snapshot.Signals...), snapshot.Projections...) {
		if input.LatencyMS != nil {
			t.Fatalf("Router response invented per-signal latency for %q", input.ID)
		}
	}
	for _, eligibility := range snapshot.Eligibility {
		if eligibility.State != "eligible" || eligibility.ReasonCode != "none" {
			t.Fatalf("recommended frozen arm was not normalized as eligible: %+v", eligibility)
		}
	}
	if err := ValidateRoutingRecipeDecisionSnapshot(manifest.Target.Mixture.RoutingRecipePlan, *snapshot); err != nil {
		t.Fatalf("normalized routing decision rejected: %v", err)
	}
	report, err := ReduceRoutingRecipeEvaluation(RoutingRecipeReductionInput{
		Plan: manifest.Target.Mixture.RoutingRecipePlan, ExpectedCaseIDs: []string{snapshot.CaseID},
		Decisions: []RoutingRecipeDecisionSnapshot{*snapshot},
		Outcomes: []RoutingRecipeOutcome{
			{DecisionID: snapshot.DecisionID, CaseID: snapshot.CaseID, ArmID: "arm-fast", ObservedAt: observedAt.Add(time.Second), Quality: 0.2},
			{DecisionID: snapshot.DecisionID, CaseID: snapshot.CaseID, ArmID: "arm-strong", ObservedAt: observedAt.Add(time.Second), Quality: 0.9},
		},
	})
	if err != nil {
		t.Fatalf("reduce ordered recommendation ranking: %v", err)
	}
	if len(report.E2.TopK) != 2 || !report.E2.TopK[0].Recall.Available || report.E2.TopK[0].K != 1 ||
		report.E2.TopK[0].Recall.Value != 0 || !report.E2.TopK[1].Recall.Available ||
		report.E2.TopK[1].K != 2 || report.E2.TopK[1].Recall.Value != 1 {
		t.Fatalf("ordered recommendations did not preserve top-k discrimination: %+v", report.E2.TopK)
	}
}

func TestRoutingRecipeBrokerMapsFallbackAbstentionAndRequestFailureFailClosed(t *testing.T) {
	manifest := routingRecipeBrokerTestManifest(t)
	request := workerBrokerRequest{ID: 2, Operation: workerBrokerRouterEvaluate, TrackID: "routing", CaseID: "case-3"}
	observedAt := time.Date(2026, 8, 31, 10, 11, 12, 0, time.UTC)

	fallbackModel, fallbackArm, fallbackStatus := "model-strong", "arm-strong", "fallback"
	fallbackEntry := executionAttestationEntry{
		RequestID: request.ID, Operation: request.Operation, TrackID: request.TrackID, CaseID: request.CaseID,
		UpstreamAttempted: true, Success: true, SelectedModel: &fallbackModel, ArmID: &fallbackArm,
		SelectionStatus: &fallbackStatus,
	}
	fallback := routingRecipeDecisionFromBrokerResponse(manifest, request, workerBrokerResponse{
		Success: true, FetchedAt: observedAt, Payload: map[string]any{"recommended_models": []any{"model-strong"}},
	}, fallbackEntry)
	if fallback == nil || fallback.SelectionStatus != "fallback" || fallback.SelectedArmID != fallbackArm ||
		len(fallback.RankedArmIDs) != 1 || fallback.RankedArmIDs[0] != fallbackArm {
		t.Fatalf("fallback normalization = %+v", fallback)
	}

	abstainedStatus := "execution_required"
	abstainedEntry := executionAttestationEntry{
		RequestID: request.ID, Operation: request.Operation, TrackID: request.TrackID, CaseID: request.CaseID,
		UpstreamAttempted: true, Success: true, SelectionStatus: &abstainedStatus,
	}
	abstained := routingRecipeDecisionFromBrokerResponse(manifest, request, workerBrokerResponse{
		Success: true, FetchedAt: observedAt, Payload: map[string]any{"recommended_models": []any{"model-fast"}},
	}, abstainedEntry)
	if abstained == nil || abstained.SelectionStatus != "abstained" || abstained.SelectedArmID != "" ||
		len(abstained.RankedArmIDs) != 1 || abstained.RankedArmIDs[0] != "arm-fast" {
		t.Fatalf("execution_required normalization = %+v", abstained)
	}

	timeout := "request_timeout"
	failedEntry := executionAttestationEntry{
		RequestID: request.ID, Operation: request.Operation, TrackID: request.TrackID, CaseID: request.CaseID,
		AttemptID: "attempt-3", UpstreamAttempted: true,
	}
	failed := routingRecipeDecisionFromBrokerResponse(manifest, request, workerBrokerResponse{
		FetchedAt: observedAt, Error: &timeout,
	}, failedEntry)
	if failed == nil || failed.SelectionStatus != "error" || failed.SelectedArmID != "" || len(failed.RankedArmIDs) != 0 {
		t.Fatalf("failed request normalization = %+v", failed)
	}
	for _, input := range append(append([]RoutingRecipeObservedInput{}, failed.Signals...), failed.Projections...) {
		if input.State != "timeout" || input.Value != nil || input.LatencyMS != nil || input.ErrorCode != "" {
			t.Fatalf("failed request input was not fail-closed: %+v", input)
		}
	}

	unattemptedEntry := failedEntry
	unattemptedEntry.UpstreamAttempted = false
	unattempted := routingRecipeDecisionFromBrokerResponse(manifest, request, workerBrokerResponse{
		FetchedAt: observedAt,
	}, unattemptedEntry)
	if unattempted == nil || unattempted.SelectionStatus != "unavailable" ||
		unattempted.SelectedArmID != "" || len(unattempted.RankedArmIDs) != 0 {
		t.Fatalf("unattempted request normalization = %+v", unattempted)
	}
	unattemptedEntry.RoutingRecipeDecision = unattempted
	unattemptedEntry.FetchedAt = &observedAt
	unattemptedEntry.RequestedModel = &manifest.Target.Mixture.EntrypointModel
	unattemptedEntry.Recipe = &manifest.Target.Mixture.RecipeName
	unattemptedEntry.RequestDigest = digestString("unattempted-request")
	unattemptedEntry.ResponseDigest = digestString("")
	unattemptedEntry.Headers = map[string]string{}
	var receiptErr error
	unattemptedEntry.BrokerReceipt, receiptErr = brokerEntryReceipt(unattemptedEntry)
	if receiptErr != nil {
		t.Fatalf("unavailable routing receipt: %v", receiptErr)
	}
	if err := validateStoredExecutionAttestationEntry(unattemptedEntry, request.ID); err != nil {
		t.Fatalf("unavailable routing decision was not persistable: %v", err)
	}
	if err := validateBrokerRoutingRecipeDecision(manifest.Target.Mixture, unattemptedEntry); err != nil {
		t.Fatalf("unavailable routing decision lost its exact manifest binding: %v", err)
	}
	discoveryStatus := http.StatusOK
	discovery := executionAttestationEntry{
		RequestID: 1, Operation: workerBrokerListModels,
		RequestDigest: digestString("models-request"), ResponseDigest: digestString("models-response"),
		UpstreamAttempted: true, Success: true, StatusCode: &discoveryStatus,
		Headers: map[string]string{},
	}
	discovery.BrokerReceipt, receiptErr = brokerEntryReceipt(discovery)
	if receiptErr != nil {
		t.Fatalf("model discovery receipt: %v", receiptErr)
	}
	if _, err := indexBrokerAttestationEntries(
		manifest,
		[]executionAttestationEntry{discovery, unattemptedEntry},
		observedAt.Add(-time.Second), observedAt.Add(time.Second),
	); err != nil {
		t.Fatalf("unavailable pre-upstream routing decision did not survive persistence binding: %v", err)
	}
}

func TestRoutingRecipeBrokerPreservesRecommendationOrderAndInfeasibleSelection(t *testing.T) {
	manifest := routingRecipeBrokerTestManifest(t)
	request := workerBrokerRequest{
		ID: 8, Operation: workerBrokerRouterEvaluate, TrackID: "routing", CaseID: "case-8",
	}
	observedAt := time.Date(2026, 8, 31, 10, 11, 12, 0, time.UTC)
	selectedModel, selectedArmID, selectionStatus := "model-strong", "arm-strong", "fallback"
	entry := executionAttestationEntry{
		RequestID: request.ID, Operation: request.Operation, TrackID: request.TrackID,
		CaseID: request.CaseID, AttemptID: "attempt-8", UpstreamAttempted: true, Success: true,
		SelectedModel: &selectedModel, ArmID: &selectedArmID, SelectionStatus: &selectionStatus,
	}
	snapshot := routingRecipeDecisionFromBrokerResponse(
		manifest, request,
		workerBrokerResponse{Success: true, FetchedAt: observedAt, Payload: map[string]any{
			"recommended_models": []any{"model-fast"},
		}},
		entry,
	)
	if snapshot == nil || snapshot.SelectionStatus != "fallback" || snapshot.SelectedArmID != "arm-strong" ||
		len(snapshot.RankedArmIDs) != 1 || snapshot.RankedArmIDs[0] != "arm-fast" {
		t.Fatalf("ordered infeasible fallback snapshot = %+v", snapshot)
	}
	states := make(map[string]string, len(snapshot.Eligibility))
	for _, item := range snapshot.Eligibility {
		states[item.ArmID] = item.State + "/" + item.ReasonCode
	}
	if states["arm-fast"] != "eligible/none" || states["arm-strong"] != "ineligible/not_recommended" {
		t.Fatalf("complete recommendation eligibility = %+v", snapshot.Eligibility)
	}
	if err := ValidateRoutingRecipeDecisionSnapshot(manifest.Target.Mixture.RoutingRecipePlan, *snapshot); err != nil {
		t.Fatalf("infeasible frozen fallback was rejected: %v", err)
	}
	entry.FetchedAt = &observedAt
	entry.RoutingRecipeDecision = snapshot
	if err := validateBrokerRoutingRecipeDecision(manifest.Target.Mixture, entry); err != nil {
		t.Fatalf("broker binding erased an infeasible frozen fallback: %v", err)
	}
	report, err := ReduceRoutingRecipeEvaluation(RoutingRecipeReductionInput{
		Plan: manifest.Target.Mixture.RoutingRecipePlan, ExpectedCaseIDs: []string{request.CaseID},
		Decisions: []RoutingRecipeDecisionSnapshot{*snapshot}, Outcomes: []RoutingRecipeOutcome{},
	})
	if err != nil {
		t.Fatalf("reduce infeasible fallback: %v", err)
	}
	if report.E1.EligibilityComplete != 1 || report.E1.SelectedFeasible != 0 {
		t.Fatalf("infeasible selection lost metric discrimination: %+v", report.E1)
	}

	selectionStatus = "selected"
	selectedModel, selectedArmID = "model-fast", "arm-fast"
	entry.SelectedModel, entry.ArmID, entry.SelectionStatus = &selectedModel, &selectedArmID, &selectionStatus
	missingRecommendations := routingRecipeDecisionFromBrokerResponse(
		manifest, request,
		workerBrokerResponse{Success: true, FetchedAt: observedAt, Payload: map[string]any{}},
		entry,
	)
	if missingRecommendations == nil || len(missingRecommendations.RankedArmIDs) != 1 ||
		missingRecommendations.RankedArmIDs[0] != selectedArmID {
		t.Fatalf("selection without recommendations did not retain its bounded first arm: %+v", missingRecommendations)
	}
	for _, item := range missingRecommendations.Eligibility {
		if item.State != "unavailable" || item.ReasonCode != routingRecipeUnavailableReason {
			t.Fatalf("selection invented eligibility without recommendations: %+v", item)
		}
	}
	if err := ValidateRoutingRecipeDecisionSnapshot(
		manifest.Target.Mixture.RoutingRecipePlan, *missingRecommendations,
	); err != nil {
		t.Fatalf("selection with unavailable recommendation evidence was rejected: %v", err)
	}
}

func TestRoutingRecipeBrokerRejectsUnknownEligibilityAndManifestMismatch(t *testing.T) {
	manifest := routingRecipeBrokerTestManifest(t)
	observedAt := time.Date(2026, 8, 31, 10, 11, 12, 0, time.UTC)
	selectedModel, selectedArmID, selectionStatus := "model-fast", "arm-fast", "selected"
	entry := executionAttestationEntry{
		RequestID: 5, Operation: workerBrokerRouterEvaluate, TrackID: "routing",
		CaseID: "case-5", AttemptID: "attempt-5", UpstreamAttempted: true, Success: true,
		FetchedAt: &observedAt, SelectedModel: &selectedModel, ArmID: &selectedArmID, SelectionStatus: &selectionStatus,
	}
	entry.RoutingRecipeDecision = routingRecipeDecisionFromBrokerResponse(
		manifest,
		workerBrokerRequest{ID: 5, Operation: workerBrokerRouterEvaluate, TrackID: "routing", CaseID: "case-5"},
		workerBrokerResponse{Success: true, FetchedAt: observedAt, Payload: map[string]any{
			"recommended_models": []any{"foreign-model"},
		}},
		entry,
	)
	if entry.RoutingRecipeDecision == nil || entry.RoutingRecipeDecision.SelectionStatus != "error" {
		t.Fatalf("foreign eligibility was not failed closed: %+v", entry.RoutingRecipeDecision)
	}
	if err := validateBrokerRoutingRecipeDecision(manifest.Target.Mixture, entry); err == nil {
		t.Fatal("foreign recommendation remained consistent with a selected response")
	}

	entry.RoutingRecipeDecision = routingRecipeDecisionFromBrokerResponse(
		manifest,
		workerBrokerRequest{ID: 5, Operation: workerBrokerRouterEvaluate, TrackID: "routing", CaseID: "case-5"},
		workerBrokerResponse{Success: true, FetchedAt: observedAt, Payload: map[string]any{
			"recommended_models": []any{"model-fast"},
		}},
		entry,
	)
	mutatedMixture := *manifest.Target.Mixture
	mutatedPlan := mutatedMixture.RoutingRecipePlan
	mutatedPlan.Signals = append(mutatedPlan.Signals, RoutingRecipeInputSpec{ID: "language:english", ValueKind: "numeric"})
	var err error
	mutatedPlan, err = canonicalRoutingRecipePlan(mutatedPlan)
	if err != nil {
		t.Fatalf("canonicalize mutated plan: %v", err)
	}
	mutatedMixture.RoutingRecipePlan = mutatedPlan
	if err := validateBrokerRoutingRecipeDecision(&mutatedMixture, entry); err == nil {
		t.Fatal("decision snapshot was accepted against a different frozen manifest plan")
	}
}

func TestRoutingRecipeDecisionMutationInvalidatesBrokerReceiptAndAttestation(t *testing.T) {
	manifest := routingRecipeBrokerTestManifest(t)
	now := time.Date(2026, 8, 31, 10, 11, 12, 0, time.UTC)
	selectedModel, selectedArmID, selectionStatus, recipe := "model-fast", "arm-fast", "selected", manifest.Target.Mixture.RecipeName
	selectionMethod, decisionName := "static", "quality"
	status := http.StatusOK
	router := executionAttestationEntry{
		RequestID: 2, Operation: workerBrokerRouterEvaluate, TrackID: "routing", CaseID: "case-2", AttemptID: "attempt-2",
		RequestDigest: digestString("routing-request"), ResponseDigest: digestString("routing-response"),
		UpstreamAttempted: true, Success: true, StatusCode: &status, LatencyMicroseconds: 10,
		FetchedAt: &now, Headers: map[string]string{}, RequestedModel: &manifest.Target.Mixture.EntrypointModel,
		SelectedModel: &selectedModel, ArmID: &selectedArmID, SelectionStatus: &selectionStatus, Recipe: &recipe,
		SelectionMethod: &selectionMethod, Algorithm: &selectionMethod, DecisionName: &decisionName,
	}
	router.RoutingRecipeDecision = routingRecipeDecisionFromBrokerResponse(
		manifest,
		workerBrokerRequest{ID: 2, Operation: workerBrokerRouterEvaluate, TrackID: "routing", CaseID: "case-2"},
		workerBrokerResponse{Success: true, FetchedAt: now, Payload: map[string]any{
			"signal_confidences": map[string]any{"domain:reasoning": json.Number("0.9"), "projection:oracle-probability": json.Number("0.8")},
			"recommended_models": []any{"model-fast"},
		}},
		router,
	)
	modelDiscovery := executionAttestationEntry{
		RequestID: 1, Operation: workerBrokerListModels,
		RequestDigest: digestString("models-request"), ResponseDigest: digestString("models-response"),
		UpstreamAttempted: true, Success: true, StatusCode: &status, LatencyMicroseconds: 5,
		FetchedAt: &now, Headers: map[string]string{},
	}
	var operationErr error
	modelDiscovery.BrokerReceipt, operationErr = brokerEntryReceipt(modelDiscovery)
	if operationErr != nil {
		t.Fatalf("models receipt: %v", operationErr)
	}
	router.BrokerReceipt, operationErr = brokerEntryReceipt(router)
	if operationErr != nil {
		t.Fatalf("routing receipt: %v", operationErr)
	}
	runID := newTestClientRequestID()
	manifest.RunID = runID
	manifest.ManifestDigest = digestString("manifest")
	manifest.Target.ID = "target"
	manifest.PolicySnapshotDigest = digestString("policy")
	manifest.Target.BackendTopologyDigest = digestString("topology")
	attestation := executionAttestation{
		SchemaVersion: SchemaVersion, ContractVersion: executionAttestationContractVersion,
		RunID: runID, ManifestDigest: manifest.ManifestDigest, TargetID: manifest.Target.ID,
		Mode: ModeLive, PolicySnapshotDigest: manifest.PolicySnapshotDigest, BackendTopologyDigest: manifest.Target.BackendTopologyDigest,
		StartedAt: now.Add(-time.Second), CompletedAt: now.Add(time.Second),
		Entries: []executionAttestationEntry{modelDiscovery, router},
	}
	attestation.Digest, operationErr = executionAttestationDigest(attestation)
	if operationErr != nil || validateExecutionAttestationIdentity(runID, attestation) != nil {
		t.Fatalf("valid attestation rejected: digestErr=%v validationErr=%v", operationErr, validateExecutionAttestationIdentity(runID, attestation))
	}
	store := newPrivateTestStore(t)
	if err := store.writeExecutionAttestation(attestation); err != nil {
		t.Fatalf("persist broker attestation: %v", err)
	}
	if _, err := store.readExecutionAttestationForManifest(runID, manifest); err != nil {
		t.Fatalf("exact manifest read rejected: %v", err)
	}
	assertRoutingRecipeManifestMutationRejected(t, store, runID, manifest, attestation)

	mutated := attestation
	mutated.Entries = append([]executionAttestationEntry(nil), attestation.Entries...)
	decision := *mutated.Entries[1].RoutingRecipeDecision
	decision.Signals = append([]RoutingRecipeObservedInput(nil), decision.Signals...)
	for index := range decision.Signals {
		if decision.Signals[index].ID == "domain:reasoning" {
			decision.Signals[index].Value = floatPointer(0.1)
		}
	}
	mutated.Entries[1].RoutingRecipeDecision = &decision
	mutated.Digest, operationErr = executionAttestationDigest(mutated)
	if operationErr != nil {
		t.Fatalf("digest mutated attestation: %v", operationErr)
	}
	if err := validateExecutionAttestationIdentity(runID, mutated); err == nil {
		t.Fatal("routing decision mutation retained its original broker receipt")
	}
}

func assertRoutingRecipeManifestMutationRejected(
	t *testing.T,
	store *Store,
	runID string,
	manifest RunManifest,
	attestation executionAttestation,
) {
	t.Helper()
	readMismatch := manifest
	readMismatch.Target.Mixture = copyManifestMixture(manifest.Target.Mixture)
	readMismatch.Target.Mixture.RoutingRecipePlan.Signals = append(
		readMismatch.Target.Mixture.RoutingRecipePlan.Signals,
		RoutingRecipeInputSpec{ID: "language:english", ValueKind: "numeric"},
	)
	canonicalPlan, operationErr := canonicalRoutingRecipePlan(
		readMismatch.Target.Mixture.RoutingRecipePlan,
	)
	if operationErr != nil {
		t.Fatalf("canonicalize read-mismatch plan: %v", operationErr)
	}
	readMismatch.Target.Mixture.RoutingRecipePlan = canonicalPlan
	if _, err := store.readExecutionAttestationForManifest(runID, readMismatch); err == nil {
		t.Fatal("durable read accepted a self-consistent attestation against a different manifest plan")
	}
	if _, err := indexBrokerAttestationEntries(
		readMismatch, attestation.Entries, attestation.StartedAt, attestation.CompletedAt,
	); err == nil {
		t.Fatal("persistence binding accepted a broker transcript against a different manifest plan")
	}
}

func TestRoutingRecipeWorkerCannotSupplyDecisionTimeOrOutcomeFields(t *testing.T) {
	manifest := routingRecipeBrokerTestManifest(t)
	broker := newWorkerHTTPBroker(manifest, workerBrokerCredentials{})
	broker.models[manifest.Target.Mixture.EntrypointModel] = manifest.Target.Mixture.RecipeName
	broker.modelsValid = true
	base := map[string]any{
		"model":                manifest.Target.Mixture.EntrypointModel,
		"messages":             []any{map[string]any{"role": "user", "content": "route"}},
		"evaluate_all_signals": true,
	}
	for _, field := range []string{"observed_at", "ranked_arm_ids", "outcomes"} {
		payload := make(map[string]any, len(base)+1)
		for key, value := range base {
			payload[key] = value
		}
		payload[field] = "worker-controlled"
		encoded, err := json.Marshal(payload)
		if err != nil {
			t.Fatalf("encode payload: %v", err)
		}
		if _, err := broker.validatedPayload(workerBrokerRouterEvaluate, encoded); err == nil {
			t.Fatalf("worker-controlled %s crossed the strict Router request contract", field)
		}
	}
}

func TestBrokerCaseRequestBindingRejectsPromptSubstitution(t *testing.T) {
	plannedMessages := []visibleMessage{{
		Role: "user", Content: json.RawMessage(`[{"type":"text","text":"planned prompt"}]`),
	}}
	// Reordered object keys must not change the semantic digest.
	equivalentMessages := []brokerMessage{{
		Role: "user", Content: json.RawMessage(`[{"text":"planned prompt","type":"text"}]`),
	}}
	plannedDigest, digestErr := canonicalMessageListDigest(plannedMessages)
	if digestErr != nil {
		t.Fatalf("digest planned messages: %v", digestErr)
	}
	equivalentDigest, digestErr := canonicalMessageListDigest(equivalentMessages)
	if digestErr != nil || equivalentDigest != plannedDigest {
		t.Fatalf("semantic message digest drifted: got=%q want=%q err=%v", equivalentDigest, plannedDigest, digestErr)
	}
	forgedDigest, digestErr := canonicalMessageListDigest([]brokerMessage{{
		Role: "user", Content: json.RawMessage(`"attacker-selected prompt"`),
	}})
	if digestErr != nil {
		t.Fatalf("digest forged messages: %v", digestErr)
	}
	model := "fixture-entrypoint"
	requestDigest, digestErr := brokerRequestDigestForMessages(workerBrokerRouterEvaluate, model, forgedDigest, 0)
	if digestErr != nil {
		t.Fatalf("digest forged Router request: %v", digestErr)
	}
	entry := executionAttestationEntry{
		Operation: workerBrokerRouterEvaluate, RequestedModel: &model, RequestDigest: requestDigest,
	}
	record := executionRecordEvidence{TrackID: "routing", CaseID: "case-1"}
	cases := visibleCaseSet{MessageDigests: map[string]string{"case-1": plannedDigest}}
	if err := validateBrokerCaseRequestBinding(entry, record, cases, 0); err == nil {
		t.Fatal("worker-selected prompt was accepted for a server-sealed case id")
	}
	entry.RequestDigest, digestErr = brokerRequestDigestForMessages(workerBrokerRouterEvaluate, model, plannedDigest, 0)
	if digestErr != nil {
		t.Fatalf("digest planned Router request: %v", digestErr)
	}
	if err := validateBrokerCaseRequestBinding(entry, record, cases, 0); err != nil {
		t.Fatalf("exact server-sealed case input rejected: %v", err)
	}
}

func routingRecipeBrokerTestManifest(t *testing.T) RunManifest {
	t.Helper()
	mixture := brokerTestMixture()
	mixture.FallbackArmID = "arm-strong"
	plan := mixture.RoutingRecipePlan
	plan.FallbackArmID = mixture.FallbackArmID
	plan.Signals = []RoutingRecipeInputSpec{
		{ID: "domain:reasoning", ValueKind: "numeric"},
		{ID: "structure:many_questions", ValueKind: "numeric"},
		{ID: "classifier:risk:RISKY", ValueKind: "numeric"},
		{ID: "language:english", ValueKind: "numeric"},
		{ID: "context:turns", ValueKind: "numeric"},
		{ID: "metadata:absent", ValueKind: "numeric"},
	}
	plan.Projections = []RoutingRecipeProjectionSpec{{
		ID: "projection:oracle-probability", ValueKind: "probability", OutcomeBinding: "selected_is_oracle",
	}}
	var err error
	plan, err = canonicalRoutingRecipePlan(plan)
	if err != nil {
		t.Fatalf("canonicalize broker routing plan: %v", err)
	}
	mixture.RoutingRecipePlan = plan
	if err := validateMixtureContract(mixture); err != nil {
		t.Fatalf("broker routing mixture contract: %v", err)
	}
	return RunManifest{
		Mode: ModeLive, TrackIDs: []TrackID{"routing"}, Concurrency: 1,
		Target: ManifestTarget{Mixture: mixture},
	}
}

func assertRoutingRecipeObservedInput(
	t *testing.T,
	inputs []RoutingRecipeObservedInput,
	id, state string,
	wantValue float64,
	wantError string,
) {
	t.Helper()
	for _, input := range inputs {
		if input.ID != id {
			continue
		}
		if input.State != state || input.ErrorCode != wantError {
			t.Fatalf("routing input %q = %+v, want state=%q error=%q", id, input, state, wantError)
		}
		if state == "present" && input.Value == nil {
			t.Fatalf("routing input %q omitted its present value", id)
		}
		if input.Value != nil && *input.Value != wantValue {
			t.Fatalf("routing input %q value=%v, want %v", id, *input.Value, wantValue)
		}
		return
	}
	t.Fatalf("routing input %q not found", id)
}
