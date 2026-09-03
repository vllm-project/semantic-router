package evaluationplane

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

func registerCustomProviderContracts(t *testing.T, registry *Registry) (string, string, string) {
	t.Helper()
	const (
		executorID = "provider-live.v1"
		suiteID    = "provider-agent-methods"
	)
	if err := registry.registerExecutor(executorContract{
		ID: executorID, Mode: ModeLive, SuiteClass: executorSuiteRuntime,
		TargetProfile: targetProfileRuntime, LineageProfile: lineageRuntime,
		TrackIDs: []TrackID{"agentic", "preference", "safety"}, EvidenceLevelCeiling: "E0",
	}); err != nil {
		t.Fatalf("register provider executor: %v", err)
	}
	healthy := true
	providerMixture := catalogTestNamedMixtureSnapshot(
		"provider-recipe",
		[]string{"provider-mom", "provider-mom-alias"},
		[]ModelArm{catalogTestArm("provider-deep", []string{"text"}), catalogTestArm("provider-fast", []string{"text"})},
		digestString("provider-topology"),
	)
	targetID := providerMixture.Mixture.ID
	if err := registry.registerTarget(targetDefinition{
		Public: CatalogTarget{
			ID: targetID, Name: "Provider runtime", Description: "Test provider-owned runtime.",
			Kind: "mixture-of-models", Modes: []Mode{ModeLive},
			AcceptedExecutors: map[Mode][]string{ModeLive: {executorID}},
			EvidenceLevel:     "E0", Healthy: &healthy,
			Mixture: catalogMixtureFromManifest(&providerMixture.Mixture),
		},
		Contract: targetContract{ExecutionProfile: targetProfileRuntime, PolicySnapshot: policySnapshotRuntime, TrackRequirements: map[TrackID][]targetFeature{
			"agentic": {"agent-runtime", targetFeatureFaultRecoveryLedger}, "preference": {targetFeatureProductionExperimentLedger}, "safety": {targetFeatureHardPolicyLedger},
		}},
		EnvoyURL:                   "http://provider-envoy.invalid",
		FaultRecoveryLedger:        &ServiceEndpoint{SchemaVersion: SchemaVersion, URL: "http://provider-recovery-ledger.invalid", TimeoutSeconds: 30},
		HardPolicyLedger:           &ServiceEndpoint{SchemaVersion: SchemaVersion, URL: "http://provider-policy-ledger.invalid", TimeoutSeconds: 30},
		ProductionExperimentLedger: &ServiceEndpoint{SchemaVersion: SchemaVersion, URL: "http://provider-experiment-ledger.invalid", TimeoutSeconds: 30},
		Mixture:                    copyManifestMixture(&providerMixture.Mixture),
		ConfigDigest:               digestString("provider-config"),
		BackendTopologyDigest:      providerMixture.BackendTopologyDigest,
		Features:                   []targetFeature{"agent-runtime"},
	}); err != nil {
		t.Fatalf("register provider target: %v", err)
	}
	if err := registry.registerSuite(CatalogSuite{
		ID: suiteID, Name: "Provider agent methods", Description: "Provider executor extension test.",
		Executors: map[Mode]string{ModeLive: executorID}, TrackIDs: []TrackID{"agentic", "preference", "safety"},
		Modes: []Mode{ModeLive}, EvidenceLevel: "E0", CaseCount: 1,
		Revision: "provider-v1", Tags: []string{"provider"}, Methods: []CatalogMethod{
			{ID: "provider.fault-recovery.v1", TrackID: "agentic", QualifiedGateIDs: []string{"G6"}, EvidenceSource: "live_runtime", Status: "configured"},
			{ID: "provider.production-experiment.v1", TrackID: "preference", QualifiedGateIDs: []string{"G8", "G9"}, EvidenceSource: "live_production", Status: "configured"},
			{ID: "provider.hard-policy.v1", TrackID: "safety", QualifiedGateIDs: []string{"G2"}, EvidenceSource: "live_runtime", Status: "configured"},
		},
	}); err != nil {
		t.Fatalf("register provider suite: %v", err)
	}
	return executorID, suiteID, targetID
}

func TestCustomProviderContractDrivesCreateManifestAndWorkerStaging(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	registry, err := service.registrySnapshot()
	if err != nil {
		t.Fatalf("registry snapshot: %v", err)
	}

	executorID, suiteID, targetID := registerCustomProviderContracts(t, registry)

	catalog := registry.Catalog()
	if !catalogContainsTarget(catalog, targetID) || !catalogContainsSuite(catalog, suiteID) {
		t.Fatalf("custom provider registration is absent from catalog: %+v", catalog)
	}
	request := CreateRunRequest{
		ClientRequestID: newTestClientRequestID(), Name: "provider agent methods",
		SuiteIDs: []string{suiteID}, TrackIDs: []TrackID{"agentic", "preference", "safety"},
		Mode: ModeLive, TargetID: targetID, ChangeProfile: "agent_multimodal",
		SampleLimit: 1, Concurrency: 1, Seed: 17,
	}
	validated, target, err := service.validateCreateRequest(registry, request)
	if err != nil {
		t.Fatalf("validate provider run: %v", err)
	}
	evidenceLevel, err := selectedSuiteEvidenceLevel(registry, validated.SuiteIDs, validated.Mode)
	if err != nil {
		t.Fatalf("resolve provider evidence level: %v", err)
	}
	run, manifest, err := service.newPendingRunManifest(registry, validated, target, evidenceLevel)
	if err != nil {
		t.Fatalf("freeze provider manifest: %v", err)
	}
	if manifest.SuiteExecutors[suiteID] != executorID || manifest.Target.ID != targetID {
		t.Fatalf("provider identity was not frozen: %+v", manifest)
	}
	if manifest.Target.FaultRecoveryLedger == nil || manifest.Target.HardPolicyLedger == nil || manifest.Target.ProductionExperimentLedger == nil {
		t.Fatalf("provider ledger capability endpoints were not frozen: %+v", manifest.Target)
	}
	if _, resolveErr := registry.executionContracts().resolve(manifest); resolveErr != nil {
		t.Fatalf("resolve provider execution contract: %v", resolveErr)
	}
	tamperedConfig := manifest
	tamperedConfig.ConfigDigest = digestString("different-provider-config")
	if _, resolveErr := registry.executionContracts().resolve(tamperedConfig); !errors.Is(resolveErr, ErrInvalid) || !strings.Contains(resolveErr.Error(), "config digest") {
		t.Fatalf("target-specific config digest error=%v, want config digest ErrInvalid", resolveErr)
	}
	if _, persistErr := service.persistPendingRunAs(SystemActor(), validated, run, manifest); persistErr != nil {
		t.Fatalf("persist provider run: %v", persistErr)
	}
	manifestPath := filepath.Join(root, "runs", run.ID, manifestFileName)
	staging, err := prepareWorkerStaging(ProcessSpec{
		ManifestPath:       manifestPath,
		StorePath:          root,
		executionContracts: registry.executionContracts(),
	})
	if err != nil {
		t.Fatalf("stage provider run: %v", err)
	}
	staging.cleanup()

	manifest.SuiteExecutors[suiteID] = "retired-provider.v0"
	manifest.ManifestDigest, err = manifestSemanticDigest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if writeErr := writeJSONAtomic(manifestPath, manifest); writeErr != nil {
		t.Fatal(writeErr)
	}
	_, err = prepareWorkerStaging(ProcessSpec{
		ManifestPath:       manifestPath,
		StorePath:          root,
		executionContracts: registry.executionContracts(),
	})
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "executor") {
		t.Fatalf("unregistered provider executor error=%v, want executor ErrInvalid", err)
	}
}

func catalogContainsTarget(catalog Catalog, targetID string) bool {
	for _, target := range catalog.Targets {
		if target.ID == targetID {
			return true
		}
	}
	return false
}

func catalogContainsSuite(catalog Catalog, suiteID string) bool {
	for _, suite := range catalog.Suites {
		if suite.ID == suiteID {
			return true
		}
	}
	return false
}
