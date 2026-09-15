package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// versionGateConformanceCase is one row of the shared cross-language corpus.
// version_present distinguishes an omitted key from an explicit value, since
// the raw gate treats them differently for anything that isn't a string.
type versionGateConformanceCase struct {
	ID             string `json:"id"`
	VersionPresent bool   `json:"version_present"`
	Version        string `json:"version"`
	ExpectedValid  bool   `json:"expected_valid"`
}

type versionGateConformanceContract struct {
	AcceptedVersions []string                     `json:"accepted_versions"`
	Cases            []versionGateConformanceCase `json:"cases"`
}

type versionGateConformanceCorpus struct {
	SchemaVersion              string                         `json:"schema_version"`
	CurrentContract            versionGateConformanceContract `json:"current_contract"`
	RetainedContractSimulation versionGateConformanceContract `json:"retained_contract_simulation"`
}

func loadVersionGateConformanceCorpus(t *testing.T) versionGateConformanceCorpus {
	t.Helper()
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve version gate conformance test location")
	}
	fixturePath := filepath.Join(
		filepath.Dir(currentFile),
		"../../../vllm-sr/tests/fixtures/router_config_version_gate_conformance.v1.json",
	)
	payload, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read shared version gate conformance corpus: %v", err)
	}
	var corpus versionGateConformanceCorpus
	if err := json.Unmarshal(payload, &corpus); err != nil {
		t.Fatalf("decode shared version gate conformance corpus: %v", err)
	}
	if corpus.SchemaVersion != "router-config-version-gate-conformance.v1" ||
		len(corpus.CurrentContract.Cases) == 0 || len(corpus.RetainedContractSimulation.Cases) == 0 {
		t.Fatalf("shared version gate conformance corpus has an invalid envelope: %#v", corpus)
	}
	return corpus
}

// runVersionGateConformanceContract asserts ValidateRawCanonicalVersion agrees
// with the corpus for one accepted_versions set, so the same fixture the CLI's
// generated schema and Pydantic model are checked against is also checked
// against the Go gate they are meant to match.
func runVersionGateConformanceContract(t *testing.T, contract versionGateConformanceContract) {
	t.Helper()
	if len(contract.AcceptedVersions) == 0 {
		t.Fatal("conformance contract has no accepted_versions")
	}
	restore := acceptedCanonicalVersions
	t.Cleanup(func() { acceptedCanonicalVersions = restore })
	acceptedCanonicalVersions = contract.AcceptedVersions

	seenIDs := make(map[string]struct{}, len(contract.Cases))
	for _, tc := range contract.Cases {
		t.Run(tc.ID, func(t *testing.T) {
			if _, duplicate := seenIDs[tc.ID]; duplicate {
				t.Fatalf("duplicate version gate conformance case id %q", tc.ID)
			}
			seenIDs[tc.ID] = struct{}{}

			raw := map[string]interface{}{}
			if tc.VersionPresent {
				raw["version"] = tc.Version
			}

			err := ValidateRawCanonicalVersion(raw)
			if (err == nil) != tc.ExpectedValid {
				t.Fatalf("ValidateRawCanonicalVersion(present=%v, version=%q) error = %v, want valid=%v",
					tc.VersionPresent, tc.Version, err, tc.ExpectedValid)
			}
		})
	}
}

func TestVersionGateMatchesSharedCrossLanguageConformance(t *testing.T) {
	corpus := loadVersionGateConformanceCorpus(t)

	t.Run("current_contract", func(t *testing.T) {
		runVersionGateConformanceContract(t, corpus.CurrentContract)
	})
	t.Run("retained_contract_simulation", func(t *testing.T) {
		runVersionGateConformanceContract(t, corpus.RetainedContractSimulation)
	})
}
