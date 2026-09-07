package evaluationplane

import (
	"errors"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestExecutionAttestationStoredContractRejectsTampering(t *testing.T) {
	runID := newTestClientRequestID()
	tests := map[string]func(*executionAttestation){
		"non-sequential request": func(value *executionAttestation) {
			value.Entries[0].RequestID = 2
		},
		"unknown operation": func(value *executionAttestation) {
			value.Entries[0].Operation = "network.proxy"
		},
		"invalid status": func(value *executionAttestation) {
			status := 700
			value.Entries[0].StatusCode = &status
		},
		"unapproved header": func(value *executionAttestation) {
			value.Entries[0].Headers["authorization"] = "secret"
		},
		"unbounded observation": func(value *executionAttestation) {
			observation := strings.Repeat("x", maxBrokerObservedFieldBytes+1)
			value.Entries[0].SelectedModel = &observation
		},
		"failed model discovery": func(value *executionAttestation) {
			value.Entries[0].Success = false
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			value := validExecutionAttestation(t, runID)
			mutate(&value)
			refreshExecutionAttestationDigests(t, &value)
			if err := validateExecutionAttestationIdentity(runID, value); err == nil {
				t.Fatal("tampered execution attestation was accepted")
			}
		})
	}

	value := validExecutionAttestation(t, runID)
	value.Entries[0].Quality = floatPointer(math.NaN())
	if err := validateStoredExecutionAttestationEntry(value.Entries[0], 1); err == nil {
		t.Fatal("non-finite execution quality was accepted")
	}
}

func TestConfidenceSnapshotSealsExactLooperAlgorithmInExecutionAttestation(t *testing.T) {
	manifest, attestation, algorithm := confidenceExecutionAttestationFixture(t)
	store := newPrivateTestStore(t)
	if err := store.writeExecutionAttestation(attestation); err != nil {
		t.Fatalf("seal confidence execution attestation: %v", err)
	}
	sealed, err := store.readExecutionAttestationForManifest(manifest.RunID, manifest)
	if err != nil {
		t.Fatalf("read confidence attestation against frozen snapshot: %v", err)
	}
	if sealed.Entries[1].Algorithm == nil || *sealed.Entries[1].Algorithm != algorithm {
		t.Fatalf("sealed attestation algorithm=%v, want %q", sealed.Entries[1].Algorithm, algorithm)
	}
}

func confidenceMixtureFixture(t *testing.T) MixtureTargetSnapshot {
	t.Helper()
	confidenceYAML := strings.Replace(
		multiRecipeMixtureTestYAML,
		`        type: prompt
        prompt: {model: selector, instructions: "Choose one candidate."}`,
		`        type: confidence
        minimum_candidates: 2
        confidence:
          confidence_method: hybrid
          threshold: 0.72
          escalation_order: small_to_large
          on_error: skip`,
		1,
	)
	if confidenceYAML == multiRecipeMixtureTestYAML {
		t.Fatal("confidence fixture did not replace the selector algorithm")
	}
	snapshot, err := ModelArmSnapshotFromYAML([]byte(confidenceYAML), "revision")
	if err != nil {
		t.Fatalf("freeze confidence target snapshot: %v", err)
	}
	target := mixtureForRecipe(t, snapshot, "default")
	if !target.Ready || len(target.Mixture.Decisions) != 1 ||
		target.Mixture.Decisions[0].Algorithm != "confidence" {
		t.Fatalf("confidence target did not preserve its Router algorithm: %#v", target)
	}
	return target
}

func confidenceExecutionAttestationFixture(t *testing.T) (RunManifest, executionAttestation, string) {
	t.Helper()
	target := confidenceMixtureFixture(t)
	mixture := target.Mixture
	decision := mixture.Decisions[0]
	selectedArmID := decision.ArmIDs[0]
	selectedModel := ""
	for _, arm := range mixture.ModelArms {
		if arm.ID == selectedArmID {
			selectedModel = arm.Model
			break
		}
	}
	if selectedModel == "" {
		t.Fatalf("confidence decision references an unknown frozen arm: %#v", decision)
	}

	runID := newTestClientRequestID()
	manifest := RunManifest{
		RunID: runID, ManifestDigest: digestString("manifest"), Mode: ModeLive,
		TrackIDs: []TrackID{"joint"}, PolicySnapshotDigest: digestString("policy"),
		Target: ManifestTarget{
			SchemaVersion: SchemaVersion, ID: mixture.ID, Kind: "mixture-of-models",
			Mixture: &mixture, BackendTopologyDigest: target.BackendTopologyDigest,
		},
	}
	now := time.Now().UTC()
	status := http.StatusOK
	selectionStatus := "selected"
	algorithm := "confidence"
	decisionName := decision.Name
	inputTokens, outputTokens := int64(10), int64(4)
	responseContentDigest := digestString("response-content")
	requestedModel := mixture.EntrypointModel
	recipe := mixture.RecipeName
	entries := []executionAttestationEntry{
		{
			RequestID: 1, Operation: workerBrokerListModels,
			RequestDigest: digestString("models-request"), ResponseDigest: digestString("models-response"),
			UpstreamAttempted: true, Success: true, StatusCode: &status,
			LatencyMicroseconds: 10, FetchedAt: &now, Headers: map[string]string{},
		},
		{
			RequestID: 2, Operation: workerBrokerRoutedChatCompletion,
			TrackID: "joint", CaseID: "case-1", AttemptID: "attempt-1",
			RequestDigest: digestString("chat-request"), ResponseDigest: digestString("chat-response"),
			UpstreamAttempted: true, Success: true, StatusCode: &status,
			LatencyMicroseconds: 20, FetchedAt: &now,
			Headers: map[string]string{
				"x-vsr-selected-model": selectedModel, "x-vsr-selected-algorithm": algorithm,
				"x-vsr-selected-recipe": recipe, "x-vsr-selected-decision": decisionName,
			},
			RequestedModel: &requestedModel, SelectedModel: &selectedModel, ArmID: &selectedArmID,
			SelectionStatus: &selectionStatus, SelectionMethod: &algorithm,
			Recipe: &recipe, DecisionName: &decisionName, Algorithm: &algorithm,
			InputTokens: &inputTokens, OutputTokens: &outputTokens,
			ResponseContentDigest: &responseContentDigest,
		},
	}
	attestation := executionAttestation{
		SchemaVersion: SchemaVersion, ContractVersion: executionAttestationContractVersion,
		RunID: runID, ManifestDigest: manifest.ManifestDigest, TargetID: manifest.Target.ID,
		Mode: ModeLive, PolicySnapshotDigest: manifest.PolicySnapshotDigest,
		BackendTopologyDigest: manifest.Target.BackendTopologyDigest,
		StartedAt:             now.Add(-time.Second), CompletedAt: now.Add(time.Second), Entries: entries,
	}
	refreshExecutionAttestationDigests(t, &attestation)
	return manifest, attestation, algorithm
}

func TestMixtureAuthorizationRejectsUnknownFallbackAlgorithm(t *testing.T) {
	mixture := brokerTestMixture()
	mixture.FallbackArmID = "arm-strong"
	armID := mixture.FallbackArmID
	status := "fallback"
	unknown := "unknown"
	decisionName := mixture.Decisions[0].Name
	if mixtureAuthorizesSelection(mixture, executionAttestationEntry{
		ArmID: &armID, SelectionStatus: &status, Algorithm: &unknown, DecisionName: &decisionName,
	}) {
		t.Fatal("unknown algorithm used fallback status as a frozen-decision wildcard")
	}
	configured := mixture.Decisions[0].Algorithm
	configuredArm := mixture.Decisions[0].ArmIDs[0]
	if mixtureAuthorizesSelection(mixture, executionAttestationEntry{
		ArmID: &configuredArm, SelectionStatus: &status, Algorithm: &configured,
	}) {
		t.Fatal("configured algorithm was authorized without its exact frozen decision")
	}
	providerDefault := "default"
	if !mixtureAuthorizesSelection(mixture, executionAttestationEntry{
		ArmID: &armID, SelectionStatus: &status, Algorithm: &providerDefault,
	}) {
		t.Fatal("provider-default runtime selection lost its frozen fallback authorization")
	}
}

func TestStoredExecutionAttestationBindsDirectArmAndNormalizedContentDigest(t *testing.T) {
	status := http.StatusOK
	requestedModel := "model-fast"
	armID := "arm-fast"
	selectedModel := "arm-fast"
	contentDigest := digestString("exact answer")
	inputTokens := int64(3)
	outputTokens := int64(2)
	entry := executionAttestationEntry{
		RequestID: 1, Operation: workerBrokerArmChatCompletion,
		TrackID: "model_pool", CaseID: "case-1", AttemptID: "attempt-arm-fast",
		RequestDigest: digestString("request"), ResponseDigest: digestString("response"),
		UpstreamAttempted: true, Success: true, StatusCode: &status,
		LatencyMicroseconds: 100, Headers: map[string]string{"x-vsr-selected-model": selectedModel},
		RequestedModel: &requestedModel, ArmID: &armID, SelectedModel: &selectedModel,
		InputTokens: &inputTokens, OutputTokens: &outputTokens, ResponseContentDigest: &contentDigest,
	}
	receipt, err := brokerEntryReceipt(entry)
	if err != nil {
		t.Fatalf("digest direct-arm receipt: %v", err)
	}
	entry.BrokerReceipt = receipt
	if err := validateStoredExecutionAttestationEntry(entry, 1); err != nil {
		t.Fatalf("valid direct-arm attestation rejected: %v", err)
	}

	invalidDigest := "raw response text"
	entry.ResponseContentDigest = &invalidDigest
	if err := validateStoredExecutionAttestationEntry(entry, 1); err == nil {
		t.Fatal("non-digest response content observation was accepted")
	}
}

func TestMixtureRecordDensityAndServerOwnedQualityCost(t *testing.T) {
	mixture := brokerTestMixture()
	manifest := RunManifest{
		TrackIDs: []TrackID{"routing", "model_pool", "joint"},
		Target:   ManifestTarget{Mixture: mixture},
	}
	cases := visibleCaseSet{CaseIDsByTrack: map[TrackID]map[string]struct{}{
		"routing":    {"case-1": {}},
		"model_pool": {"case-1": {}},
		"joint":      {"case-1": {}},
	}}
	armFast := "arm-fast"
	armStrong := "arm-strong"
	records := []executionRecordEvidence{
		{ID: "pool-fast", TrackID: "model_pool", CaseID: "case-1", ArmID: &armFast},
		{ID: "pool-strong", TrackID: "model_pool", CaseID: "case-1", ArmID: &armStrong},
		{ID: "joint", TrackID: "joint", CaseID: "case-1", SelectedArmID: &armFast},
	}
	if err := validateMixtureRecordDensity(manifest, records, cases); err != nil {
		t.Fatalf("dense frozen mixture matrix rejected: %v", err)
	}
	for name, mutated := range map[string][]executionRecordEvidence{
		"missing arm":     records[1:],
		"duplicate arm":   append(append([]executionRecordEvidence(nil), records...), records[0]),
		"duplicate joint": append(append([]executionRecordEvidence(nil), records...), records[2]),
	} {
		if err := validateMixtureRecordDensity(manifest, mutated, cases); err == nil {
			t.Fatalf("%s matrix was accepted", name)
		}
	}

	expectedAnswer := " exact\nanswer "
	contentDigest := digestString("exact answer")
	inputTokens := int64(3)
	outputTokens := int64(2)
	entry := executionAttestationEntry{
		Success: true, ArmID: &armFast, ResponseContentDigest: &contentDigest,
		InputTokens: &inputTokens, OutputTokens: &outputTokens,
	}
	quality := serverObservedQuality(entry, "model_pool", gradingCaseEvidence{ExpectedAnswer: &expectedAnswer}, nil)
	if quality == nil || *quality != 1 {
		t.Fatalf("server normalized exact-match quality = %v, want 1", quality)
	}
	caseMismatchedAnswer := " exact\nANSWER "
	caseMismatched := serverObservedQuality(
		entry, "model_pool", gradingCaseEvidence{ExpectedAnswer: &caseMismatchedAnswer}, nil,
	)
	if caseMismatched == nil || *caseMismatched != 0 {
		t.Fatalf("case-mismatched exact answer quality = %v, want 0", caseMismatched)
	}
	cost := serverRuntimeCost(entry, mixture.ModelArms)
	wantCost := (3*mixture.ModelArms[0].InputCostPerMillionTokensUSD +
		2*mixture.ModelArms[0].OutputCostPerMillionTokensUSD) / 1_000_000
	if cost == nil || *cost != wantCost {
		t.Fatalf("server runtime cost = %v, want %g", cost, wantCost)
	}

	wrongDigest := digestString("wrong answer")
	entries := []executionAttestationEntry{
		{Operation: workerBrokerArmChatCompletion, TrackID: "model_pool", CaseID: "case-1", ArmID: &armFast, Success: true, ResponseContentDigest: &contentDigest, BrokerReceipt: "receipt-fast"},
		{Operation: workerBrokerArmChatCompletion, TrackID: "model_pool", CaseID: "case-1", ArmID: &armStrong, Success: true, ResponseContentDigest: &wrongDigest, BrokerReceipt: "receipt-strong"},
	}
	recordsByReceipt := map[string][]executionRecordEvidence{
		"receipt-fast":   {{TrackID: "model_pool", CaseID: "case-1", ArmID: &armFast}},
		"receipt-strong": {{TrackID: "model_pool", CaseID: "case-1", ArmID: &armStrong}},
	}
	grading := map[string]gradingCaseEvidence{"case-1": {ExpectedAnswer: &expectedAnswer}}
	oracles := serverPoolOracleArmIDs(manifest, entries, recordsByReceipt, cases, grading)
	if _, fastIsOracle := oracles["case-1"][armFast]; !fastIsOracle || len(oracles["case-1"]) != 1 {
		t.Fatalf("server pool oracle arms = %#v, want only %q", oracles, armFast)
	}
	routingEntry := executionAttestationEntry{Success: true, ArmID: &armFast}
	routingQuality := serverObservedQuality(routingEntry, "routing", grading["case-1"], oracles["case-1"])
	if routingQuality == nil || *routingQuality != 1 {
		t.Fatalf("oracle-derived routing quality = %v, want 1", routingQuality)
	}
	missingOracle := serverPoolOracleArmIDs(manifest, entries[:1], recordsByReceipt, cases, grading)
	if quality := serverObservedQuality(routingEntry, "routing", grading["case-1"], missingOracle["case-1"]); quality != nil {
		t.Fatalf("incomplete pool evidence produced routing quality %v", quality)
	}
}

func TestNewStoreRecoversOnlyUnanchoredExecutionAttestations(t *testing.T) {
	t.Run("orphan after run deletion", func(t *testing.T) {
		store := newPrivateTestStore(t)
		runID := newTestClientRequestID()
		if err := store.writeExecutionAttestation(validExecutionAttestation(t, runID)); err != nil {
			t.Fatalf("write orphan attestation: %v", err)
		}
		path := filepath.Join(store.attestationRoot, runID+".json")
		if _, err := newStandaloneStore(store.root); err != nil {
			t.Fatalf("recover orphan attestation: %v", err)
		}
		if _, err := os.Lstat(path); !os.IsNotExist(err) {
			t.Fatalf("orphan execution attestation survived recovery: %v", err)
		}
	})

	t.Run("published before report anchor", func(t *testing.T) {
		store := newPrivateTestStore(t)
		runID := newTestClientRequestID()
		makePrivateRunDirectory(t, store, runID)
		if err := store.writeExecutionAttestation(validExecutionAttestation(t, runID)); err != nil {
			t.Fatalf("write unanchored attestation: %v", err)
		}
		if _, err := newStandaloneStore(store.root); err != nil {
			t.Fatalf("recover unanchored attestation: %v", err)
		}
		if _, err := os.Lstat(filepath.Join(store.attestationRoot, runID+".json")); !os.IsNotExist(err) {
			t.Fatalf("unanchored execution attestation survived recovery: %v", err)
		}
	})

	t.Run("report anchor without exact manifest", func(t *testing.T) {
		store := newPrivateTestStore(t)
		runID := newTestClientRequestID()
		makePrivateRunDirectory(t, store, runID)
		attestation := validExecutionAttestation(t, runID)
		if err := store.writeExecutionAttestation(attestation); err != nil {
			t.Fatalf("write anchored attestation: %v", err)
		}
		if err := store.writeReportAnchor(runID, testExecutionReportAnchor(runID, attestation.Digest)); err != nil {
			t.Fatalf("write matching report anchor: %v", err)
		}
		if _, err := newStandaloneStore(store.root); err != nil {
			t.Fatalf("restart store with anchored attestation: %v", err)
		}
		if _, err := store.readExecutionAttestation(runID); !errors.Is(err, ErrNotFound) {
			t.Fatalf("attestation without its exact manifest survived recovery: %v", err)
		}
	})
}

func TestExecutionAttestationLifecycleRejectsUnsafeEntriesAndDeletesWithRun(t *testing.T) {
	store := newPrivateTestStore(t)
	invalidName := filepath.Join(store.attestationRoot, "not-a-run.json")
	if err := os.WriteFile(invalidName, []byte("{}\n"), 0o600); err != nil {
		t.Fatalf("write invalid attestation entry: %v", err)
	}
	if _, err := newStandaloneStore(store.root); err == nil {
		t.Fatal("store startup silently removed an unsafe attestation entry")
	}
	if _, err := os.Lstat(invalidName); err != nil {
		t.Fatalf("unsafe attestation entry was modified: %v", err)
	}
	if err := os.Remove(invalidName); err != nil {
		t.Fatalf("remove test entry: %v", err)
	}

	runID := newTestClientRequestID()
	makePrivateRunDirectory(t, store, runID)
	if err := store.writeExecutionAttestation(validExecutionAttestation(t, runID)); err != nil {
		t.Fatalf("write execution attestation: %v", err)
	}
	if err := store.DeleteRunAs(SystemActor(), runID); err != nil {
		t.Fatalf("delete run with execution attestation: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(store.attestationRoot, runID+".json")); !os.IsNotExist(err) {
		t.Fatalf("run deletion left its execution attestation behind: %v", err)
	}
}

func validExecutionAttestation(t *testing.T, runID string) executionAttestation {
	t.Helper()
	status := 200
	entry := executionAttestationEntry{
		RequestID: 1, Operation: workerBrokerListModels,
		RequestDigest: digestBytes(nil), ResponseDigest: digestString(`{"data":[]}`),
		UpstreamAttempted: true, Success: true, StatusCode: &status,
		LatencyMicroseconds: 100, Headers: map[string]string{},
	}
	receipt, err := brokerEntryReceipt(entry)
	if err != nil {
		t.Fatalf("digest broker entry: %v", err)
	}
	entry.BrokerReceipt = receipt
	now := time.Now().UTC()
	value := executionAttestation{
		SchemaVersion: SchemaVersion, ContractVersion: executionAttestationContractVersion,
		RunID: runID, ManifestDigest: digestString("manifest"), TargetID: "mom-default", Mode: ModeLive,
		PolicySnapshotDigest: digestString("policy"), BackendTopologyDigest: digestString("topology"),
		StartedAt: now.Add(-time.Second), CompletedAt: now, Entries: []executionAttestationEntry{entry},
	}
	value.Digest, err = executionAttestationDigest(value)
	if err != nil {
		t.Fatalf("digest execution attestation: %v", err)
	}
	return value
}

// validExecutionAttestationForManifest builds the shared durable fixture from
// the immutable manifest instead of retrofitting only its top-level digests.
// Routing runs therefore carry the same broker-owned, plan-bound decision
// snapshot required of production evidence.
func validExecutionAttestationForManifest(t *testing.T, manifest RunManifest) executionAttestation {
	t.Helper()
	value := validExecutionAttestation(t, manifest.RunID)
	value.ManifestDigest = manifest.ManifestDigest
	value.TargetID = manifest.Target.ID
	value.Mode = manifest.Mode
	value.PolicySnapshotDigest = manifest.PolicySnapshotDigest
	value.BackendTopologyDigest = manifest.Target.BackendTopologyDigest
	if manifest.Mode == ModeLive && containsTrack(manifest.TrackIDs, "routing") {
		if manifest.Target.Mixture == nil {
			t.Fatal("routing attestation fixture requires a frozen mixture")
		}
		observedAt := value.CompletedAt
		requestedModel := manifest.Target.Mixture.EntrypointModel
		recipe := manifest.Target.Mixture.RecipeName
		entry := executionAttestationEntry{
			RequestID: 2, Operation: workerBrokerRouterEvaluate, TrackID: "routing",
			CaseID: "case-1", AttemptID: "attempt-1",
			RequestDigest: digestString("routing-request"), ResponseDigest: digestString(""),
			UpstreamAttempted: true, LatencyMicroseconds: 100, FetchedAt: &observedAt,
			Headers: map[string]string{}, RequestedModel: &requestedModel, Recipe: &recipe,
		}
		requestError := "request_error"
		entry.RoutingRecipeDecision = routingRecipeDecisionFromBrokerResponse(
			manifest,
			workerBrokerRequest{
				ID: entry.RequestID, Operation: entry.Operation, TrackID: entry.TrackID,
				CaseID: entry.CaseID, AttemptID: entry.AttemptID,
			},
			workerBrokerResponse{FetchedAt: observedAt, Error: &requestError},
			entry,
		)
		if entry.RoutingRecipeDecision == nil {
			t.Fatal("routing attestation fixture did not create a broker-owned decision snapshot")
		}
		value.Entries = append(value.Entries, entry)
	}
	refreshExecutionAttestationDigests(t, &value)
	if err := validateExecutionAttestationIdentity(manifest.RunID, value); err != nil {
		t.Fatalf("manifest-aware execution attestation identity: %v", err)
	}
	if err := validateExecutionAttestationAgainstManifest(value, manifest); err != nil {
		t.Fatalf("manifest-aware execution attestation binding: %v", err)
	}
	return value
}

func refreshExecutionAttestationDigests(t *testing.T, value *executionAttestation) {
	t.Helper()
	for index := range value.Entries {
		value.Entries[index].BrokerReceipt = ""
		receipt, err := brokerEntryReceipt(value.Entries[index])
		if err != nil {
			t.Fatalf("refresh broker receipt: %v", err)
		}
		value.Entries[index].BrokerReceipt = receipt
	}
	value.Digest = ""
	digest, err := executionAttestationDigest(*value)
	if err != nil {
		t.Fatalf("refresh execution attestation digest: %v", err)
	}
	value.Digest = digest
}

func newPrivateTestStore(t *testing.T) *Store {
	t.Helper()
	root := t.TempDir()
	if err := os.Chmod(root, 0o700); err != nil {
		t.Fatalf("protect test store: %v", err)
	}
	store, err := newStandaloneStore(root)
	if err != nil {
		t.Fatalf("create test store: %v", err)
	}
	return store
}

func makePrivateRunDirectory(t *testing.T, store *Store, runID string) {
	t.Helper()
	runDir := filepath.Join(store.runsRoot, runID)
	if err := os.Mkdir(runDir, 0o700); err != nil {
		t.Fatalf("create private run directory: %v", err)
	}
	mixture := brokerTestMixture()
	run := Run{
		SchemaVersion: SchemaVersion, ID: runID, ClientRequestID: runID,
		Name: "execution attestation fixture", Status: StatusPending, Mode: ModeLive, EvidenceLevel: "E0",
		TargetID: mixture.ID, Mixture: catalogMixtureFromManifest(mixture),
		ChangeProfile: "schema_adapter", SuiteIDs: []string{"evaluation-smoke"},
		TrackIDs: []TrackID{"routing"}, TrackEvidenceLevels: map[TrackID]EvidenceLevel{"routing": "E0"},
		SampleLimit: 1, Concurrency: 1, Seed: 17,
		Progress:  RunProgress{Total: 1, Message: "Evaluation pending"},
		CreatedAt: time.Now().UTC().Truncate(time.Microsecond),
	}
	if err := writeJSONAtomic(filepath.Join(runDir, runFileName), run); err != nil {
		t.Fatalf("write private run status: %v", err)
	}
	actor := SystemActor()
	lifecycle := newRunLifecycle(run, actor)
	store.lifecycle.mu.Lock()
	audit, err := store.appendLifecycleAuditUnlocked(
		actor, lifecycleResourceRun, "create", "allowed", "system", run.ID, actor.PrincipalDigest(),
	)
	store.lifecycle.mu.Unlock()
	if err != nil {
		t.Fatalf("write lifecycle creation audit: %v", err)
	}
	lifecycle.CreationAuditDigest = audit.Digest
	lifecycle.PolicyDigest = lifecycleDigest(lifecycle)
	if err := writeJSONAtomic(filepath.Join(runDir, lifecycleFileName), lifecycle); err != nil {
		t.Fatalf("write private run lifecycle: %v", err)
	}
}

func testExecutionReportAnchor(runID, attestationDigest string) reportAnchor {
	return reportAnchor{
		SchemaVersion: SchemaVersion, AttestationRevision: ServerAttestationRevision,
		RunID: runID, ReportDigest: digestString("report"), ReportSize: 1,
		ManifestSemanticDigest:     digestString("manifest-semantic"),
		ManifestArtifactDigest:     digestString("manifest-artifact"),
		PrivateReceiptDigest:       digestString("receipt"),
		ExecutionAttestationDigest: attestationDigest,
		EvidenceFiles: []sealedEvidenceFile{{
			Scope: "run", Name: "report.json", Digest: digestString("report"),
			SizeBytes: 1,
		}},
		CreatedAt: time.Now().UTC(),
	}
}
