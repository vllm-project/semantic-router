package handlers

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type dashboardContractCorpus struct {
	SupportedVersion string                        `json:"supported_version"`
	SteadyState      []dashboardContractSteadyCase `json:"steady_state"`
}

type dashboardContractSteadyCase struct {
	Name              string `json:"name"`
	Input             string `json:"input"`
	Valid             bool   `json:"valid"`
	Error             string `json:"error"`
	NormalizedVersion string `json:"normalized_version"`
}

func TestDashboardExecutesCanonicalContractGoldenCorpus(t *testing.T) {
	t.Parallel()

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test source path")
	}
	corpusPath := filepath.Join(
		filepath.Dir(filename),
		"..", "..", "..",
		"src", "semantic-router", "pkg", "config", "testdata",
		"canonical_contract_cases.json",
	)
	data, err := os.ReadFile(corpusPath)
	if err != nil {
		t.Fatalf("read canonical contract corpus: %v", err)
	}
	var corpus dashboardContractCorpus
	if err := json.Unmarshal(data, &corpus); err != nil {
		t.Fatalf("decode canonical contract corpus: %v", err)
	}
	if corpus.SupportedVersion != routerconfig.CanonicalVersion {
		t.Fatalf("corpus version = %q, contract version = %q", corpus.SupportedVersion, routerconfig.CanonicalVersion)
	}

	for _, test := range corpus.SteadyState {
		t.Run(test.Name, func(t *testing.T) {
			t.Parallel()

			path := filepath.Join(t.TempDir(), "config.yaml")
			if err := os.WriteFile(path, []byte(test.Input), 0o600); err != nil {
				t.Fatalf("write config fixture: %v", err)
			}
			cfg, err := readCanonicalConfigFile(path)
			if !test.Valid {
				if err == nil {
					t.Fatal("expected dashboard contract rejection")
				}
				if !strings.Contains(err.Error(), test.Error) {
					t.Fatalf("expected %q in error, got %v", test.Error, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("expected dashboard contract acceptance, got %v", err)
			}
			if cfg.Version != test.NormalizedVersion {
				t.Fatalf("normalized version = %q, want %q", cfg.Version, test.NormalizedVersion)
			}
		})
	}
}

func TestDashboardSetupFileUsesCanonicalVersionAndUnknownFieldContract(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input string
		error string
	}{
		{
			name:  "missing version",
			input: "setup:\n  mode: true\n",
			error: "version: required",
		},
		{
			name:  "future version",
			input: "version: v99.0\nsetup:\n  mode: true\n",
			error: "v99.0",
		},
		{
			name:  "unknown setup field",
			input: "version: v0.3\nsetup:\n  mode: true\n  modde: false\n",
			error: "setup.modde",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			path := filepath.Join(t.TempDir(), "config.yaml")
			if err := os.WriteFile(path, []byte(test.input), 0o600); err != nil {
				t.Fatalf("write setup fixture: %v", err)
			}
			_, err := readSetupConfigFile(path)
			if err == nil || !strings.Contains(err.Error(), test.error) {
				t.Fatalf("expected %q in error, got %v", test.error, err)
			}
		})
	}
}
