package evaluationplane

import (
	"encoding/json"
	"testing"
)

func builtinExecutorContractForTest(t *testing.T, id string) executorContract {
	t.Helper()
	for _, contract := range builtinExecutorContracts() {
		if contract.ID == id {
			return contract
		}
	}
	t.Fatalf("builtin executor contract %q is not registered", id)
	return executorContract{}
}

func manifestExecutorContractForTest(t *testing.T, manifest RunManifest) executorContract {
	t.Helper()
	id, ok := manifestExecutorIdentity(manifest)
	if !ok {
		t.Fatal("manifest has no single executor identity")
	}
	return builtinExecutorContractForTest(t, id)
}

func serviceExecutionContractsForTest(t *testing.T, service *Service) *executionContractRegistry {
	t.Helper()
	registry, err := service.registrySnapshot()
	if err != nil {
		t.Fatalf("registry snapshot: %v", err)
	}
	return registry.executionContracts()
}

func validateRecordsAndFailureSummaryForTest(
	t *testing.T,
	runDir string,
	manifest RunManifest,
) (recordAttestation, error) {
	t.Helper()
	return validateRecordsAndFailureSummary(runDir, manifest, manifestExecutorContractForTest(t, manifest))
}

func resolveSuiteGateQualificationForTest(
	t *testing.T,
	root string,
	manifest RunManifest,
) (suiteGateQualification, error) {
	t.Helper()
	return resolveSuiteGateQualification(root, manifest, manifestExecutorContractForTest(t, manifest))
}

func validateNormalizedSuiteLineageForTest(
	t *testing.T,
	runDir string,
	manifest RunManifest,
	raw json.RawMessage,
) (*normalizedSuiteIdentityLineage, error) {
	t.Helper()
	return validateNormalizedSuiteLineage(runDir, manifest, raw, manifestExecutorContractForTest(t, manifest))
}

func deriveSealedEvidenceLevelsForTest(
	t *testing.T,
	runDir string,
	manifest RunManifest,
	records recordAttestation,
	qualification suiteGateQualification,
) (sealedEvidenceLevels, error) {
	t.Helper()
	return deriveSealedEvidenceLevels(
		runDir,
		manifest,
		records,
		qualification,
		manifestExecutorContractForTest(t, manifest),
		nil,
		nil,
	)
}
