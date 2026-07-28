package config

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

type canonicalContractCorpus struct {
	SupportedVersion string                        `json:"supported_version"`
	SteadyState      []canonicalContractSteadyCase `json:"steady_state"`
}

type canonicalContractSteadyCase struct {
	Name              string `json:"name"`
	Input             string `json:"input"`
	Valid             bool   `json:"valid"`
	Error             string `json:"error"`
	NormalizedVersion string `json:"normalized_version"`
}

func loadCanonicalContractCorpus(t *testing.T) canonicalContractCorpus {
	t.Helper()

	data, err := os.ReadFile("testdata/canonical_contract_cases.json")
	if err != nil {
		t.Fatalf("read canonical contract corpus: %v", err)
	}
	var corpus canonicalContractCorpus
	if err := json.Unmarshal(data, &corpus); err != nil {
		t.Fatalf("decode canonical contract corpus: %v", err)
	}
	return corpus
}

func TestCanonicalContractGoldenCorpus(t *testing.T) {
	t.Parallel()

	corpus := loadCanonicalContractCorpus(t)
	if corpus.SupportedVersion != CanonicalVersion {
		t.Fatalf("corpus version = %q, contract version = %q", corpus.SupportedVersion, CanonicalVersion)
	}

	for _, test := range corpus.SteadyState {
		t.Run(test.Name, func(t *testing.T) {
			t.Parallel()

			cfg, err := ParseYAMLBytes([]byte(test.Input))
			if !test.Valid {
				if err == nil {
					t.Fatal("expected contract rejection")
				}
				if !strings.Contains(err.Error(), test.Error) {
					t.Fatalf("expected %q in error, got %v", test.Error, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("expected contract acceptance, got %v", err)
			}
			normalized := CanonicalConfigFromRouterConfig(cfg)
			if normalized.Version != test.NormalizedVersion {
				t.Fatalf("normalized version = %q, want %q", normalized.Version, test.NormalizedVersion)
			}
		})
	}
}

func TestParseYAMLBytesEnforcesCanonicalVersionBeforeInterpretation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		document   string
		wantDetail string
	}{
		{
			name:       "missing",
			document:   "routing: {}\n",
			wantDetail: "version: required",
		},
		{
			name:       "empty",
			document:   "version: \"\"\nrouting: {}\n",
			wantDetail: "version: must not be empty",
		},
		{
			name:       "malformed",
			document:   "version: 3\nrouting: {}\n",
			wantDetail: "version: must be a string",
		},
		{
			name:       "old without migration",
			document:   "version: v0.1\nrouting: {}\n",
			wantDetail: `unsupported config version "v0.1"`,
		},
		{
			name:       "future",
			document:   "version: v99.0\nrouting: {}\n",
			wantDetail: `unsupported config version "v99.0"`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := ParseYAMLBytes([]byte(test.document))
			if err == nil {
				t.Fatal("expected version contract to reject document")
			}
			if !strings.Contains(err.Error(), test.wantDetail) {
				t.Fatalf("expected %q in error, got %v", test.wantDetail, err)
			}
			if !strings.Contains(err.Error(), CanonicalVersion) {
				t.Fatalf("expected supported version %q in error, got %v", CanonicalVersion, err)
			}
		})
	}
}

func TestParseYAMLBytesRejectsUnknownFieldsWithIndexedPaths(t *testing.T) {
	t.Parallel()

	document := []byte(`
version: v0.3
routing:
  modelCards:
    - name: demo
      descriptino: silently-dropped-before
`)

	_, err := ParseYAMLBytes(document)
	if err == nil {
		t.Fatal("expected nested unknown field to be rejected")
	}
	for _, detail := range []string{
		"unsupported config fields",
		"routing.modelCards[0].descriptino",
		`did you mean "description"`,
	} {
		if !strings.Contains(err.Error(), detail) {
			t.Fatalf("expected %q in error, got %v", detail, err)
		}
	}
}

func TestParseYAMLBytesAllowsNamedStructuredPayloadExtension(t *testing.T) {
	t.Parallel()

	document := []byte(`
version: v0.3
routing:
  decisions:
    - name: extension
      description: named plugin extension
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs: []
      plugins:
        - type: rag
          configuration:
            enabled: true
            backend: mcp
            backend_config:
              server_name: docs
              tool_name: search
              tool_arguments:
                custom_filter:
                  nested: true
`)

	if _, err := ParseYAMLBytes(document); err != nil {
		t.Fatalf("expected named StructuredPayload extension to remain accepted, got %v", err)
	}
}

func TestUnknownFieldValidationKeepsEmptyStructsClosed(t *testing.T) {
	t.Parallel()

	type closedConfig struct{}
	err := RejectUnknownConfigValue(
		map[string]interface{}{"unexpected": true},
		reflect.TypeOf(closedConfig{}),
		"closed",
	)
	if err == nil || !strings.Contains(err.Error(), "closed.unexpected") {
		t.Fatalf("expected empty typed object to remain closed, got %v", err)
	}
}

func TestParseYAMLBytesRejectsUnknownPluginConfigurationFields(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		pluginYAML string
		wantPath   string
	}{
		{
			name: "plugin payload",
			pluginYAML: `
        - type: system_prompt
          configuration:
            system_promt: typo
`,
			wantPath: `routing.decisions[extension].plugins[0].configuration.system_promt`,
		},
		{
			name: "discriminator-owned backend payload",
			pluginYAML: `
        - type: rag
          configuration:
            enabled: true
            backend: mcp
            backend_config:
              server_nam: docs
              tool_name: search
`,
			wantPath: `routing.decisions[extension].plugins[0].configuration.backend_config.server_nam`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			document := `
version: v0.3
routing:
  decisions:
    - name: extension
      rules:
        operator: AND
        conditions: []
      plugins:
` + test.pluginYAML

			_, err := ParseYAMLBytes([]byte(document))
			if err == nil {
				t.Fatal("expected plugin configuration typo to be rejected")
			}
			if !strings.Contains(err.Error(), test.wantPath) {
				t.Fatalf("expected field path %q in error, got %v", test.wantPath, err)
			}
		})
	}
}

func TestEverySupportedDecisionPluginOwnsConfigurationValidation(t *testing.T) {
	t.Parallel()

	for _, pluginType := range SupportedDecisionPluginTypes() {
		if _, ok := decisionPluginConfigurationValidators[pluginType]; !ok {
			t.Errorf("supported plugin %q has no configuration validator", pluginType)
		}
	}
	for pluginType := range decisionPluginConfigurationValidators {
		if !IsSupportedDecisionPluginType(pluginType) {
			t.Errorf("configuration validator registered for unsupported plugin %q", pluginType)
		}
	}
}

func TestCanonicalExportUsesOwnedVersion(t *testing.T) {
	t.Parallel()

	if got := CanonicalConfigFromRouterConfig(nil).Version; got != CanonicalVersion {
		t.Fatalf("canonical export version = %q, want %q", got, CanonicalVersion)
	}
}
