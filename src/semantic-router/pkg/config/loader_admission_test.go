package config

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"
)

const admissionTestYAMLTemplate = `
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: qwen2.5:3b
  models:
    - name: qwen2.5:3b
      provider_model_id: served-qwen
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:11434
          protocol: http
routing:
  modelCards:
    - name: qwen2.5:3b
      param_size: 3b
global:
  model_catalog:
    admission:
%s
`

func admissionTestYAML(block string) []byte {
	return []byte(fmt.Sprintf(admissionTestYAMLTemplate, block))
}

func TestParseYAMLBytesParsesModelCatalogAdmission(t *testing.T) {
	cfg, err := ParseYAMLBytes(admissionTestYAML(`
      prompt_guard:
        max_concurrency: 4
        max_queue: 64
        queue_timeout_ms: 250
        on_overflow: shed
      pii_classifier:
        max_concurrency: 2
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}

	want := AdmissionConfig{MaxConcurrency: 4, MaxQueue: 64, QueueTimeoutMs: 250, OnOverflow: "shed"}
	if got := cfg.ModelAdmission["prompt_guard"]; got != want {
		t.Fatalf("prompt_guard admission = %+v, want %+v", got, want)
	}
	if got := cfg.ModelAdmission["pii_classifier"]; got != (AdmissionConfig{MaxConcurrency: 2}) {
		t.Fatalf("pii_classifier admission = %+v", got)
	}

	exported := canonicalModelCatalogFromRouterConfig(cfg)
	if got := exported.Admission["prompt_guard"]; got != want {
		t.Fatalf("exported admission = %+v, want %+v", got, want)
	}
}

func TestParseYAMLBytesRejectsInvalidAdmission(t *testing.T) {
	tests := []struct {
		name    string
		block   string
		wantErr string
	}{
		{
			"unknown deployment",
			"      unknown_model:\n        max_concurrency: 4\n",
			"unknown deployment",
		},
		{
			"missing concurrency",
			"      prompt_guard:\n        max_queue: 64\n",
			"max_concurrency must be >= 1",
		},
		{
			"invalid overflow",
			"      prompt_guard:\n        max_concurrency: 4\n        on_overflow: drop\n",
			"on_overflow must be shed, wait, or fail_open",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ParseYAMLBytes(admissionTestYAML(test.block))
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestReferenceConfigParsesAdmission(t *testing.T) {
	cfg, err := Parse(filepath.Join(referenceConfigRepoRoot(t), "config", "config.yaml"))
	if err != nil {
		t.Fatalf("reference config parse failed: %v", err)
	}
	admission, ok := cfg.ModelAdmission["prompt_guard"]
	if !ok || admission.MaxConcurrency != 4 {
		t.Fatalf("reference admission = %+v (present=%v)", admission, ok)
	}
}
