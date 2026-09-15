package evaluationplane

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

const agentTaskMixtureSnapshotGolden = "sha256:8d229b7c78bbf7865ae1b4c3dd9f6709d6afa36cbb1118274302cf03b23021d3"

func TestAgentTaskMixtureBindingMatchesPythonGolden(t *testing.T) {
	manifest := RunManifest{Target: ManifestTarget{Mixture: &ManifestMixture{
		SchemaVersion:        SchemaVersion,
		ID:                   "mom-3c013dbd4aa40261e683d162d7189e1e8238737ab4bc162464b90714666c7162",
		EntrypointModel:      "agent-entrypoint",
		Aliases:              []string{"agent-entrypoint"},
		RecipeName:           "agent-task-recipe",
		RecipeDescription:    "Frozen provider-observed agent-task subject",
		RecipeDigest:         "sha256:4d3605bae296d56eb92ebd2d28e673dbc5a68047ab2bf3f91ccf3896e4a5da20",
		PoolDigest:           "sha256:9152bdd77a5c8c36e4fffcf4f35cf9faa98d241d3807f1b98909238b7fbeb3ef",
		SelectorPolicyDigest: "sha256:76561d800b63ed6e652c5c77a42a659474e0244f5df9bd7feff9e3f4836d90e4",
		SelectorDigest:       "sha256:bb044e19b55ac2da1b884e36d1f69e395a18e3702b856fd5d5b9017f21b1005b",
		AdaptationDigest:     "sha256:ba1ad88d97808a0639a18c4fd338e83da18cb7564522c90c564c130a6f150b76",
		BindingDigest:        "sha256:dcaf1173fd3088b75345d561ff36c070f1f6959fa7fec51095d4a09d24284c89",
		ModelArms: []ModelArm{{
			ID: "agent-arm", Model: "provider/agent-model",
			ProviderModelIDDigest:        "sha256:98cc4352bb4a4e6098ac5848e2f7185dab05c426bffd7dbabb553587a6d882d1",
			InputCostPerMillionTokensUSD: 1, OutputCostPerMillionTokensUSD: 2,
		}},
		SupportModels: []SupportModel{}, FallbackArmID: "agent-arm",
		Decisions: []MixtureDecisionBinding{{Name: "default", Algorithm: "single", ArmIDs: []string{"agent-arm"}}},
	}}}
	mustFreezeTestRoutingRecipePlan(manifest.Target.Mixture)
	binding, err := methodManifestMixtureBinding(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if binding.SnapshotDigest != agentTaskMixtureSnapshotGolden {
		t.Fatalf("Mixture snapshot digest = %s, want Python golden %s", binding.SnapshotDigest, agentTaskMixtureSnapshotGolden)
	}
}

func agentTaskTestManifest(t *testing.T) RunManifest {
	t.Helper()
	manifest := RunManifest{
		Mode: ModeLive, TrackIDs: []TrackID{"agentic"},
		PolicySnapshotDigest: methodTestDigest("a"), ConfigDigest: methodTestDigest("b"),
		Target: ManifestTarget{
			ID: "mom-agent-test", Kind: "mixture-of-models", BackendTopologyDigest: methodTestDigest("c"),
			Mixture: &ManifestMixture{
				ID: "mom-agent-test", EntrypointModel: "agent-entry", Aliases: []string{"agent-entry"},
				RecipeName: "agent-recipe", RecipeDigest: methodTestDigest("d"), PoolDigest: methodTestDigest("e"),
				SelectorPolicyDigest: methodTestDigest("f"), SelectorDigest: methodTestDigest("1"),
				AdaptationDigest: methodTestDigest("2"), BindingDigest: methodTestDigest("3"),
				ModelArms: []ModelArm{
					{ID: "model-a", Model: "model-a", InputCostPerMillionTokensUSD: 100, OutputCostPerMillionTokensUSD: 500},
					{ID: "model-b", Model: "model-b", InputCostPerMillionTokensUSD: 200, OutputCostPerMillionTokensUSD: 600},
				},
			},
		},
	}
	mustFreezeTestRoutingRecipePlan(manifest.Target.Mixture)
	if _, err := methodManifestMixtureBinding(manifest); err != nil {
		t.Fatalf("mixture binding: %v", err)
	}
	return manifest
}

func agentTaskTestRows(t *testing.T) (RunManifest, agentTaskLedgerPayload, []executionRecordEvidence) {
	t.Helper()
	manifest := agentTaskTestManifest(t)
	mixture, err := methodManifestMixtureBinding(manifest)
	if err != nil {
		t.Fatal(err)
	}
	startedAt := time.Date(2026, 8, 20, 0, 0, 0, 0, time.UTC)
	attemptCount := minimumAgentTaskDistinctTaskCount * minimumAgentTaskAttemptsPerTask
	payload := agentTaskLedgerPayload{
		ContractVersion: agentTaskLedgerContractVersion, LedgerID: "agent-task-ledger", SourceID: "agent-runtime",
		Environment: "production", SuiteID: "provider-agent-tasks", SuiteRevision: methodTestDigest("4"),
		BenchmarkParityClaim: agentTaskBenchmarkParityClaim, ExecutionSemantics: agentTaskExecutionSemantics,
		ProviderAttestationDigest: methodTestDigest("5"), PolicySnapshotDigest: manifest.PolicySnapshotDigest,
		ConfigDigest: manifest.ConfigDigest, TargetID: manifest.Target.ID,
		BackendTopologyDigest: manifest.Target.BackendTopologyDigest, Mixture: mixture,
		LedgerTotalAttemptCount: attemptCount, LedgerTotalDistinctTaskCount: minimumAgentTaskDistinctTaskCount,
		MinimumDistinctTaskCount: minimumAgentTaskDistinctTaskCount, MinimumAttemptsPerTask: minimumAgentTaskAttemptsPerTask,
		WindowStartedAt: startedAt, WindowEndedAt: startedAt.Add(2 * time.Hour), SealedAt: startedAt.Add(3 * time.Hour),
	}
	records := make([]executionRecordEvidence, 0, attemptCount)
	for index := range attemptCount {
		taskIndex := index / minimumAgentTaskAttemptsPerTask
		repetition := index % minimumAgentTaskAttemptsPerTask
		attemptStartedAt := startedAt.Add(time.Duration(index+1) * time.Minute)
		toolStartedAt := attemptStartedAt.Add(5 * time.Second)
		toolCompletedAt := attemptStartedAt.Add(10 * time.Second)
		resultDigest := methodTestRowDigest(index, "tool-result")
		toolReceipt := methodTestRowDigest(index, "tool-receipt")
		method := agentTaskMethodEvidence{
			ContractVersion: agentTaskAttemptContractVersion, MethodID: agentTaskMethodID,
			LedgerID: payload.LedgerID, SourceID: payload.SourceID, SuiteID: payload.SuiteID, SuiteRevision: payload.SuiteRevision,
			BenchmarkParityClaim: payload.BenchmarkParityClaim, ExecutionSemantics: payload.ExecutionSemantics,
			PolicySnapshotDigest: payload.PolicySnapshotDigest, ConfigDigest: payload.ConfigDigest,
			TargetID: payload.TargetID, BackendTopologyDigest: payload.BackendTopologyDigest,
			MixtureSnapshotDigest: payload.Mixture.SnapshotDigest, LedgerTotalAttemptCount: attemptCount,
			LedgerTotalDistinctTaskCount: minimumAgentTaskDistinctTaskCount,
			MinimumDistinctTaskCount:     minimumAgentTaskDistinctTaskCount, MinimumAttemptsPerTask: minimumAgentTaskAttemptsPerTask,
			TaskID: "task-" + methodTestIndex(taskIndex), TaskSpecDigest: methodTestRowDigest(taskIndex, "task-spec"),
			ToolPolicy: agentTaskToolPolicy{RequiresTools: true, ExpectedTools: []string{"search"}},
			AttemptID:  "attempt-" + methodTestIndex(index), RepetitionID: "repeat-" + methodTestIndex(repetition),
			TrajectoryID: "trajectory-" + methodTestIndex(index), Seed: int64(repetition), SelectedArmID: "model-a",
			TaskSuccess: true, TaskScore: 1, SuccessThreshold: 0.5, GraderID: "provider-grader",
			GraderRevisionDigest: methodTestDigest("6"), GradingReceiptDigest: methodTestRowDigest(index, "grading"),
			PrivacyAuditReceiptDigest: methodTestRowDigest(index, "privacy"),
			ExecutionReceiptDigest:    methodTestRowDigest(index, "execution"), TrajectorySteps: 4,
			ToolCallCount: 1, InvalidToolCallCount: 0, PrivacyExposureCount: 0, InputTokens: 100, OutputTokens: 20,
			ModelCostUSD: 0.02, ToolCostUSD: 0.01, EvaluationCostUSD: 0.03, TotalCostUSD: 0.06,
			StartedAt: attemptStartedAt, CompletedAt: attemptStartedAt.Add(30 * time.Second),
			GradedAt: attemptStartedAt.Add(40 * time.Second), PrivacyAuditedAt: attemptStartedAt.Add(45 * time.Second),
			ToolCalls: []agentTaskToolCallEvidence{{
				Sequence: 1, ToolCallID: "tool-call-" + methodTestIndex(index), ToolName: "search",
				ArgumentsDigest: methodTestRowDigest(index, "arguments"), Outcome: "executed",
				ResultDigest: &resultDigest, ExecutionReceiptDigest: &toolReceipt, CostUSD: 0.01,
				StartedAt: &toolStartedAt, CompletedAt: &toolCompletedAt,
			}},
		}
		payload.Attempts = append(payload.Attempts, method)
	}
	payload.TaskSetDigest, err = agentTaskSetDigest(payload.Attempts)
	if err != nil {
		t.Fatal(err)
	}
	for index := range payload.Attempts {
		method := &payload.Attempts[index]
		method.TaskSetDigest = payload.TaskSetDigest
		caseID := methodLedgerCaseID("agent-task", payload.LedgerID, method.AttemptID)
		success := method.TaskSuccess
		quality := method.TaskScore
		selectedArm := method.SelectedArmID
		steps, toolCalls := method.TrajectorySteps, method.ToolCallCount
		invalidCalls, privacy := method.InvalidToolCallCount, method.PrivacyExposureCount
		inputTokens, outputTokens := method.InputTokens, method.OutputTokens
		runtimeCost := method.ModelCostUSD + method.ToolCostUSD
		evaluationCost := method.EvaluationCostUSD
		grader := method.GraderID
		evidenceKind := agentTaskEvidenceKind
		brokerReceipt := methodTestDigest("9")
		records = append(records, executionRecordEvidence{
			SchemaVersion: SchemaVersion, ID: "agentic-" + caseID, TrackID: "agentic", CaseID: caseID,
			AttemptID: "agentic-" + caseID, Status: "succeeded", SelectedArmID: &selectedArm,
			Success: &success, Quality: &quality, InputTokens: &inputTokens, OutputTokens: &outputTokens,
			RuntimeCost: &runtimeCost, EvaluationCost: &evaluationCost, TrajectorySteps: &steps,
			ToolCalls: &toolCalls, InvalidToolCalls: &invalidCalls, PrivacyViolations: &privacy,
			AgentTask: method, Grader: &grader, EvidenceKind: &evidenceKind, BrokerReceipt: &brokerReceipt,
		})
	}
	return manifest, payload, records
}

func TestAgentTaskLedgerAcceptsCompleteSealedRealToolCohort(t *testing.T) {
	manifest, payload, records := agentTaskTestRows(t)
	for index, record := range records {
		if err := validateAgentTaskMethod(*record.AgentTask, record); err != nil {
			t.Fatalf("attempt %d rejected: %v", index, err)
		}
	}
	if err := validateAgentTaskLedgerPayload(payload, records, manifest); err != nil {
		t.Fatalf("complete agent-task ledger rejected: %v", err)
	}
	entry := executionAttestationEntry{
		Operation: workerBrokerAgentTaskLedger, TrackID: "agentic", CaseID: "agent-task-ledger", AttemptID: "ledger-fetch",
		UpstreamAttempted: true, Success: true, BrokerReceipt: *records[0].BrokerReceipt,
		responsePayload: methodTestPayloadMap(t, payload),
	}
	entry.FetchedAt = copyTime(&payload.SealedAt)
	entry.LedgerSealedAt = copyTime(&payload.SealedAt)
	if err := validateMethodLedgerBrokerBinding(entry, records, manifest); err != nil {
		t.Fatalf("retained agent-task broker response rejected: %v", err)
	}
	reduced, err := reduceAgentTaskMethod(records)
	if err != nil {
		t.Fatal(err)
	}
	if !reduced.Complete || reduced.AttemptCount != len(records) || reduced.DistinctTaskCount != minimumAgentTaskDistinctTaskCount ||
		reduced.TaskReliability == nil || *reduced.TaskReliability != 1 || reduced.TotalCostUSD == nil ||
		reduced.ToolRequiredAttemptCount != len(records) || reduced.PureReasoningAttemptCount != 0 ||
		reduced.RequiredToolReceiptCoverage == nil || *reduced.RequiredToolReceiptCoverage != 1 {
		t.Fatalf("agent-task reduction = %+v", reduced)
	}
	levels := sealedEvidenceLevels{Run: "E0", ByTrack: map[TrackID]EvidenceLevel{"agentic": "E0"}}
	deriveLiveMethodEvidenceLevels(
		&levels,
		manifest,
		recordAttestation{Methods: methodRecordAttestation{AgentTask: reduced}},
	)
	if levels.ByTrack["agentic"] != "E5" {
		t.Fatalf("complete agent-task evidence level = %s, want E5", levels.ByTrack["agentic"])
	}
}

func TestAgentTaskLedgerRejectsAdversarialSubstitutionAndFakeExecution(t *testing.T) {
	manifest, payload, records := agentTaskTestRows(t)
	assertAgentTaskLedgerSubstitutionsRejected(t, manifest, payload, records)
	assertAgentTaskExecutionSubstitutionsRejected(t, manifest, payload, records)
}

func assertAgentTaskLedgerSubstitutionsRejected(
	t *testing.T,
	manifest RunManifest,
	payload agentTaskLedgerPayload,
	records []executionRecordEvidence,
) {
	t.Helper()
	t.Run("unknown JSON field", func(t *testing.T) {
		encoded, err := json.Marshal(payload)
		if err != nil {
			t.Fatal(err)
		}
		encoded = bytes.Replace(encoded, []byte("{"), []byte(`{"unknown":true,`), 1)
		var decoded agentTaskLedgerPayload
		if err := decodeExactJSON(encoded, &decoded); err == nil || !strings.Contains(err.Error(), "unknown") {
			t.Fatalf("unknown ledger field error = %v", err)
		}
	})

	t.Run("native parity claim", func(t *testing.T) {
		forged := payload
		forged.BenchmarkParityClaim = "native"
		if err := validateAgentTaskLedgerPayload(forged, records, manifest); err == nil {
			t.Fatal("native benchmark parity claim passed")
		}
	})

	t.Run("Mixture snapshot swap", func(t *testing.T) {
		forged := payload
		forged.Mixture.RecipeDigest = methodTestDigest("0")
		if err := validateAgentTaskLedgerPayload(forged, records, manifest); err == nil {
			t.Fatal("swapped Mixture snapshot passed")
		}
	})

	t.Run("target swap", func(t *testing.T) {
		forged := payload
		forged.TargetID = "different-target"
		if err := validateAgentTaskLedgerPayload(forged, records, manifest); err == nil {
			t.Fatal("swapped agent-task target passed")
		}
	})

	t.Run("backend topology swap", func(t *testing.T) {
		forged := payload
		forged.BackendTopologyDigest = methodTestDigest("0")
		if err := validateAgentTaskLedgerPayload(forged, records, manifest); err == nil {
			t.Fatal("swapped agent-task backend topology passed")
		}
	})

	t.Run("truncated membership", func(t *testing.T) {
		if err := validateAgentTaskLedgerPayload(payload, records[:len(records)-1], manifest); err == nil {
			t.Fatal("truncated agent-task membership passed")
		}
	})

	t.Run("reused receipt", func(t *testing.T) {
		forged := payload
		forged.Attempts = append([]agentTaskMethodEvidence(nil), payload.Attempts...)
		forged.Attempts[1].ExecutionReceiptDigest = forged.Attempts[0].ExecutionReceiptDigest
		if err := validateAgentTaskLedgerPayload(forged, records, manifest); err == nil {
			t.Fatal("reused attempt receipt passed")
		}
	})
}

func assertAgentTaskExecutionSubstitutionsRejected(
	t *testing.T,
	manifest RunManifest,
	payload agentTaskLedgerPayload,
	records []executionRecordEvidence,
) {
	t.Helper()
	t.Run("invalid tool call claims execution", func(t *testing.T) {
		forgedMethod := *records[0].AgentTask
		forgedMethod.ToolCalls = append([]agentTaskToolCallEvidence(nil), forgedMethod.ToolCalls...)
		forgedMethod.ToolCalls[0].Outcome = "rejected_invalid"
		forgedMethod.InvalidToolCallCount = 1
		forgedRecord := records[0]
		forgedRecord.AgentTask = &forgedMethod
		forgedRecord.InvalidToolCalls = &forgedMethod.InvalidToolCallCount
		if err := validateAgentTaskMethod(forgedMethod, forgedRecord); err == nil {
			t.Fatal("invalid tool call with execution receipt passed")
		}
	})

	t.Run("required attempt omits provider-executed receipt", func(t *testing.T) {
		forgedMethod := *records[0].AgentTask
		forgedMethod.ToolCalls = []agentTaskToolCallEvidence{}
		forgedMethod.ToolCallCount = 0
		forgedMethod.ToolCostUSD = 0
		forgedMethod.TotalCostUSD -= 0.01
		forgedRecord := records[0]
		forgedRecord.AgentTask = &forgedMethod
		forgedRecord.ToolCalls = &forgedMethod.ToolCallCount
		runtimeCost := forgedMethod.ModelCostUSD
		forgedRecord.RuntimeCost = &runtimeCost
		if err := validateAgentTaskMethod(forgedMethod, forgedRecord); err == nil ||
			!strings.Contains(err.Error(), "provider-executed") {
			t.Fatalf("tool-required attempt without receipt error = %v", err)
		}
	})

	t.Run("executed tool is outside expected policy", func(t *testing.T) {
		forgedMethod := *records[0].AgentTask
		forgedMethod.ToolCalls = append([]agentTaskToolCallEvidence(nil), forgedMethod.ToolCalls...)
		forgedMethod.ToolCalls[0].ToolName = "unexpected-tool"
		forgedRecord := records[0]
		forgedRecord.AgentTask = &forgedMethod
		if err := validateAgentTaskMethod(forgedMethod, forgedRecord); err == nil ||
			!strings.Contains(err.Error(), "expected-tool policy") {
			t.Fatalf("unexpected executed tool error = %v", err)
		}
	})

	t.Run("repetition changes task spec", func(t *testing.T) {
		forged := payload
		forged.Attempts = append([]agentTaskMethodEvidence(nil), payload.Attempts...)
		forged.Attempts[1].TaskSpecDigest = methodTestDigest("0")
		if err := validateAgentTaskLedgerPayload(forged, records, manifest); err == nil {
			t.Fatal("repetition task-spec drift passed")
		}
	})

	t.Run("model cost ignores frozen Mixture pricing", func(t *testing.T) {
		forged := payload
		forged.Attempts = append([]agentTaskMethodEvidence(nil), payload.Attempts...)
		forgedMethod := forged.Attempts[0]
		forgedMethod.ModelCostUSD++
		forgedMethod.TotalCostUSD++
		forged.Attempts[0] = forgedMethod
		forgedRecords := append([]executionRecordEvidence(nil), records...)
		forgedRecord := forgedRecords[0]
		forgedRecord.AgentTask = &forged.Attempts[0]
		runtimeCost := forgedMethod.ModelCostUSD + forgedMethod.ToolCostUSD
		forgedRecord.RuntimeCost = &runtimeCost
		forgedRecords[0] = forgedRecord
		if err := validateAgentTaskLedgerPayload(forged, forgedRecords, manifest); err == nil ||
			!strings.Contains(err.Error(), "frozen Mixture pricing") {
			t.Fatalf("forged model cost error = %v", err)
		}
	})
}

func TestAgentTaskPureReasoningCohortIsExplicitAndComplete(t *testing.T) {
	manifest, payload, records := agentTaskTestRows(t)
	payload.Attempts = append([]agentTaskMethodEvidence(nil), payload.Attempts...)
	rewrittenRecords := append([]executionRecordEvidence(nil), records...)
	for index := range payload.Attempts {
		method := payload.Attempts[index]
		method.ToolPolicy = agentTaskToolPolicy{RequiresTools: false, ExpectedTools: []string{}}
		method.ToolCalls = []agentTaskToolCallEvidence{}
		method.ToolCallCount = 0
		method.ToolCostUSD = 0
		method.TotalCostUSD -= 0.01
		payload.Attempts[index] = method
		record := rewrittenRecords[index]
		record.AgentTask = &payload.Attempts[index]
		record.ToolCalls = &payload.Attempts[index].ToolCallCount
		runtimeCost := payload.Attempts[index].ModelCostUSD
		record.RuntimeCost = &runtimeCost
		rewrittenRecords[index] = record
	}
	taskSetDigest, err := agentTaskSetDigest(payload.Attempts)
	if err != nil {
		t.Fatal(err)
	}
	payload.TaskSetDigest = taskSetDigest
	for index := range payload.Attempts {
		payload.Attempts[index].TaskSetDigest = taskSetDigest
		rewrittenRecords[index].AgentTask = &payload.Attempts[index]
	}
	if validateLedgerErr := validateAgentTaskLedgerPayload(payload, rewrittenRecords, manifest); validateLedgerErr != nil {
		t.Fatalf("explicit pure-reasoning cohort rejected: %v", validateLedgerErr)
	}
	reduced, err := reduceAgentTaskMethod(rewrittenRecords)
	if err != nil {
		t.Fatal(err)
	}
	if !reduced.Complete || reduced.ToolRequiredAttemptCount != 0 ||
		reduced.PureReasoningAttemptCount != len(rewrittenRecords) || reduced.RequiredToolReceiptCoverage != nil {
		t.Fatalf("pure-reasoning reduction = %+v", reduced)
	}
}

func TestAgentTaskReducerMeasuresTaskLevelReliabilitySeparatelyFromAttempts(t *testing.T) {
	_, _, records := agentTaskTestRows(t)
	failedMethod := *records[0].AgentTask
	failedMethod.TaskSuccess = false
	failedMethod.TaskScore = 0
	failedRecord := records[0]
	failedRecord.AgentTask = &failedMethod
	failedRecord.Status = "failed"
	failedRecord.Success = &failedMethod.TaskSuccess
	failedRecord.Quality = &failedMethod.TaskScore
	records[0] = failedRecord
	reduced, err := reduceAgentTaskMethod(records)
	if err != nil {
		t.Fatal(err)
	}
	if reduced.TaskSuccessRate == nil || *reduced.TaskSuccessRate != float64(len(records)-1)/float64(len(records)) ||
		reduced.TaskReliability == nil || *reduced.TaskReliability != float64(minimumAgentTaskDistinctTaskCount-1)/minimumAgentTaskDistinctTaskCount {
		t.Fatalf("attempt/task reliability conflated: %+v", reduced)
	}
}

func TestAgentTaskMetricsDoNotConflateFaultRecoveryContinuity(t *testing.T) {
	_, _, taskRecords := agentTaskTestRows(t)
	metricReducer := newRecordMetricReducer()
	for _, record := range taskRecords {
		if observeTaskErr := metricReducer.observe(record); observeTaskErr != nil {
			t.Fatal(observeTaskErr)
		}
	}
	metrics, err := metricReducer.finalize()
	if err != nil {
		t.Fatal(err)
	}
	if metrics.AgenticSuccessRate.SampleCount != len(taskRecords) || metrics.AgenticSuccessRate.Value == nil ||
		*metrics.AgenticSuccessRate.Value != 1 {
		t.Fatalf("agent-task success reduction = %+v", metrics.AgenticSuccessRate)
	}

	_, recoveryRecords := methodTestRecoveryRows(minimumRecoveryPairCount)
	metricReducer = newRecordMetricReducer()
	for _, record := range recoveryRecords {
		if observeRecoveryErr := metricReducer.observe(record); observeRecoveryErr != nil {
			t.Fatal(observeRecoveryErr)
		}
	}
	metrics, err = metricReducer.finalize()
	if err != nil {
		t.Fatal(err)
	}
	if metrics.AgenticSuccessRate.SampleCount != 0 || metrics.AgenticSuccessRate.Value != nil {
		t.Fatalf("G6 recovery rows leaked into task success: %+v", metrics.AgenticSuccessRate)
	}
}

func TestAgentTaskBrokerAttestationIsLedgerFetchNotToolExecution(t *testing.T) {
	status := 200
	fetchedAt := time.Now().UTC()
	ledgerSealedAt := fetchedAt.Add(-time.Minute)
	entry := executionAttestationEntry{
		RequestID: 1, Operation: workerBrokerAgentTaskLedger,
		TrackID: "agentic", CaseID: "agent-task-ledger", AttemptID: "ledger-fetch",
		RequestDigest: digestString("agent-task-request"), ResponseDigest: digestString("agent-task-response"),
		UpstreamAttempted: true, Success: true, StatusCode: &status,
		LatencyMicroseconds: 100, Headers: map[string]string{},
		FetchedAt: &fetchedAt, LedgerSealedAt: &ledgerSealedAt,
	}
	receipt, err := brokerEntryReceipt(entry)
	if err != nil {
		t.Fatal(err)
	}
	entry.BrokerReceipt = receipt
	if validateAttestationErr := validateStoredExecutionAttestationEntry(entry, 1); validateAttestationErr != nil {
		t.Fatalf("provider ledger fetch attestation rejected: %v", validateAttestationErr)
	}

	// The Dashboard attests only its GET of the sealed provider ledger. Model
	// or tool execution observations would overstate what this broker performed.
	requestedModel := "model-a"
	entry.RequestedModel = &requestedModel
	entry.BrokerReceipt = ""
	entry.BrokerReceipt, err = brokerEntryReceipt(entry)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateStoredExecutionAttestationEntry(entry, 1); err == nil {
		t.Fatal("agent-task ledger broker falsely claimed a model/tool execution")
	}
}

func TestAgentTaskAndFaultRecoverySuitesRequireDistinctEndpoints(t *testing.T) {
	agentTask := &ServiceEndpoint{SchemaVersion: SchemaVersion, URL: "https://agent-task.invalid", TimeoutSeconds: 30}
	recovery := &ServiceEndpoint{SchemaVersion: SchemaVersion, URL: "https://recovery.invalid", TimeoutSeconds: 30}
	if err := validateAgenticSuiteEndpoints(ModeLive, []string{"live-agent-tasks"}, agentTask, nil); err != nil {
		t.Fatalf("agent-task suite rejected its dedicated endpoint: %v", err)
	}
	if err := validateAgenticSuiteEndpoints(ModeLive, []string{"live-fault-recovery"}, nil, recovery); err != nil {
		t.Fatalf("fault-recovery suite rejected its dedicated endpoint: %v", err)
	}
	if err := validateAgenticSuiteEndpoints(ModeLive, []string{"live-agent-tasks"}, nil, recovery); err == nil ||
		!strings.Contains(err.Error(), "agent_task_ledger") {
		t.Fatalf("fault-recovery endpoint substituted for task evidence: %v", err)
	}
	if err := validateAgenticSuiteEndpoints(ModeLive, []string{"live-fault-recovery"}, agentTask, nil); err == nil ||
		!strings.Contains(err.Error(), "fault_recovery_ledger") {
		t.Fatalf("agent-task endpoint substituted for G6 recovery: %v", err)
	}
}

func TestAgentTaskCatalogMethodDoesNotQualifyG6(t *testing.T) {
	suites := builtinSuitesFor(RegistryOptions{AgentTaskLedger: &ServiceEndpoint{
		SchemaVersion: SchemaVersion, URL: "https://agent-task.invalid", TimeoutSeconds: 30,
	}})
	var taskMethod, recoveryMethod *CatalogMethod
	for suiteIndex := range suites {
		suite := &suites[suiteIndex]
		if len(suite.Methods) != 1 {
			continue
		}
		switch suite.ID {
		case "live-agent-tasks":
			taskMethod = &suite.Methods[0]
		case "live-fault-recovery":
			recoveryMethod = &suite.Methods[0]
		}
	}
	if taskMethod == nil || taskMethod.Status != "configured" || len(taskMethod.QualifiedGateIDs) != 0 {
		t.Fatalf("agent-task catalog method = %+v", taskMethod)
	}
	if recoveryMethod == nil || recoveryMethod.Status != "data_required" ||
		!reflect.DeepEqual(recoveryMethod.QualifiedGateIDs, []string{"G6"}) {
		t.Fatalf("fault-recovery catalog method = %+v", recoveryMethod)
	}
}
