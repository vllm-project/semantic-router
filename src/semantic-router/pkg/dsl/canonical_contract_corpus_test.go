package dsl

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type dslContractCorpus struct {
	SupportedVersion string                  `json:"supported_version"`
	SteadyState      []dslContractSteadyCase `json:"steady_state"`
}

type dslContractSteadyCase struct {
	Name              string `json:"name"`
	Input             string `json:"input"`
	Valid             bool   `json:"valid"`
	Error             string `json:"error"`
	NormalizedVersion string `json:"normalized_version"`
}

func TestDSLExecutesCanonicalContractGoldenCorpus(t *testing.T) {
	t.Parallel()

	corpus := loadDSLContractCorpus(t)
	if corpus.SupportedVersion != config.CanonicalVersion {
		t.Fatalf("corpus version = %q, contract version = %q", corpus.SupportedVersion, config.CanonicalVersion)
	}

	for _, test := range corpus.SteadyState {
		t.Run(test.Name, func(t *testing.T) {
			t.Parallel()
			assertDSLContractCase(t, test)
		})
	}
}

func loadDSLContractCorpus(t *testing.T) dslContractCorpus {
	t.Helper()

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test source path")
	}
	data, err := os.ReadFile(filepath.Join(
		filepath.Dir(filename),
		"..", "config", "testdata", "canonical_contract_cases.json",
	))
	if err != nil {
		t.Fatalf("read canonical contract corpus: %v", err)
	}
	var corpus dslContractCorpus
	if err := json.Unmarshal(data, &corpus); err != nil {
		t.Fatalf("decode canonical contract corpus: %v", err)
	}
	return corpus
}

func assertDSLContractCase(t *testing.T, test dslContractSteadyCase) {
	t.Helper()

	merged, err := MergeRoutingIntoBase(&config.RouterConfig{}, []byte(test.Input))
	if !test.Valid {
		if err == nil {
			t.Fatal("expected DSL contract rejection")
		}
		if !strings.Contains(err.Error(), test.Error) {
			t.Fatalf("expected %q in error, got %v", test.Error, err)
		}
		return
	}
	if err != nil {
		t.Fatalf("expected DSL contract acceptance, got %v", err)
	}
	var normalized config.CanonicalConfig
	if err := yaml.Unmarshal(merged, &normalized); err != nil {
		t.Fatalf("decode DSL output: %v", err)
	}
	if normalized.Version != test.NormalizedVersion {
		t.Fatalf("normalized version = %q, want %q", normalized.Version, test.NormalizedVersion)
	}
}
