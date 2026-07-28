package k8s

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type crdContractCorpus struct {
	SupportedVersion string                  `json:"supported_version"`
	SteadyState      []crdContractSteadyCase `json:"steady_state"`
}

type crdContractSteadyCase struct {
	Name              string `json:"name"`
	Input             string `json:"input"`
	Valid             bool   `json:"valid"`
	Error             string `json:"error"`
	NormalizedVersion string `json:"normalized_version"`
}

func TestDynamicCRDsExecuteCanonicalContractGoldenCorpus(t *testing.T) {
	t.Parallel()

	corpus := loadCRDContractCorpus(t)
	if corpus.SupportedVersion != config.CanonicalVersion {
		t.Fatalf("corpus version = %q, contract version = %q", corpus.SupportedVersion, config.CanonicalVersion)
	}

	converter := NewCRDConverter()
	for _, test := range corpus.SteadyState {
		t.Run(test.Name, func(t *testing.T) {
			t.Parallel()
			assertCRDContractCase(t, converter, test)
		})
	}
}

func loadCRDContractCorpus(t *testing.T) crdContractCorpus {
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
	var corpus crdContractCorpus
	if err := json.Unmarshal(data, &corpus); err != nil {
		t.Fatalf("decode canonical contract corpus: %v", err)
	}
	return corpus
}

func assertCRDContractCase(t *testing.T, converter *CRDConverter, test crdContractSteadyCase) {
	t.Helper()

	base, err := config.ParseYAMLBytes([]byte(test.Input))
	if !test.Valid {
		if err == nil {
			t.Fatal("expected CRD base contract rejection")
		}
		if !strings.Contains(err.Error(), test.Error) {
			t.Fatalf("expected %q in error, got %v", test.Error, err)
		}
		return
	}
	if err != nil {
		t.Fatalf("expected CRD base contract acceptance, got %v", err)
	}

	normalized, err := converter.Convert(
		&v1alpha1.IntelligentPool{},
		&v1alpha1.IntelligentRoute{},
		ptrCanonicalConfig(config.CanonicalStaticConfigFromRouterConfig(base)),
	)
	if err != nil {
		t.Fatalf("convert IntelligentPool and IntelligentRoute: %v", err)
	}
	if normalized.Version != test.NormalizedVersion {
		t.Fatalf("normalized version = %q, want %q", normalized.Version, test.NormalizedVersion)
	}
}

func ptrCanonicalConfig(cfg config.CanonicalConfig) *config.CanonicalConfig {
	return &cfg
}
