//go:build !windows

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const modalityValidationConfig = `
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: text-model
  models:
    - name: text-model
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: text-model
      modality: ar
  signals:
    modality:
      - name: AR
        description: Text-only requests.
      - name: BOTH
        description: Text and image requests.
  decisions:
    - name: image-gen
      rules:
        operator: AND
        conditions:
          - type: modality
            name: CONDITION
      modelRefs:
        - model: text-model
`

func runValidation(t *testing.T, document string) (int, configValidationReport) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	status := validateConfigFile(path, config.GatewayStandalone, false, &out)
	var report configValidationReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("stdout is not one JSON report: %q", out.String())
	}
	return status, report
}

// -validate-config refuses what the Router refuses at startup, in the
// Router's words, and exits 1.
func TestValidateConfigRefusesWhatTheRouterRefuses(t *testing.T) {
	status, report := runValidation(t, strings.Replace(modalityValidationConfig, "CONDITION", "BOTH", 1))
	if status != 1 || report.Valid {
		t.Fatalf("status %d, report %+v: a BOTH decision without a diffusion or omni model must not validate", status, report)
	}
	if !strings.Contains(report.Error, `uses modality condition "BOTH"`) {
		t.Fatalf("error %q is not the Router's", report.Error)
	}
}

// A document that loads exits 0 and carries the warnings the Router logs.
func TestValidateConfigReportsTheRoutersWarnings(t *testing.T) {
	status, report := runValidation(t, strings.Replace(modalityValidationConfig, "CONDITION", "AR", 1))
	if status != 0 || !report.Valid || report.Error != "" {
		t.Fatalf("status %d, report %+v: the document loads", status, report)
	}
	if len(report.Warnings) != 1 || report.Warnings[0].Code != "modality_detector_disabled" {
		t.Fatalf("warnings = %+v, want modality_detector_disabled", report.Warnings)
	}
}

func TestValidateConfigReportsAnUnreadableFile(t *testing.T) {
	var out bytes.Buffer
	if status := validateConfigFile(filepath.Join(t.TempDir(), "missing.yaml"), config.GatewayExtProc, false, &out); status != 1 {
		t.Fatalf("status = %d for a missing file", status)
	}
	if !strings.Contains(out.String(), `"valid":false`) {
		t.Fatalf("report = %s", out.String())
	}
}

// -migrate-config writes the deprecated model fields in the forms that
// replace them; the migrated file loads with no deprecated field.
func TestMigrateConfigRewritesDeprecatedModelFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	document := `version: v0.3
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, base_url: "http://127.0.0.1:8000/v1", provider: vllm, weight: 1}]
routing:
  signals:
    pii: [{name: personal, threshold: 0.5}]
  decisions:
    - name: route
      priority: 100
      rules: {operator: OR, conditions: [{type: pii, name: personal}]}
      modelRefs: [{model: general}]
global:
  model_catalog:
    modules:
      classifier:
        pii: {model_id: models/Vela-1.0-Encoder-307M-PII, use_cpu: true}
`
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if status := validateConfigFile(path, config.GatewayStandalone, true, &out); status != 0 {
		t.Fatalf("status %d: %s", status, out.String())
	}
	var report configValidationReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Migrated == "" || report.MigrationError != "" {
		t.Fatalf("migration report %+v", report)
	}
	for _, note := range report.Kept {
		if note.Code != "migration_change" {
			t.Fatalf("a deprecated field stays or needs an action: %+v", note)
		}
	}
	migrated, err := config.ParseYAMLBytesWithoutEnvExpansion([]byte(report.Migrated))
	if err != nil {
		t.Fatalf("the migrated file does not load: %v\n%s", err, report.Migrated)
	}
	for _, warning := range config.Warnings(migrated) {
		if warning.Code == "deprecated_model_field" {
			t.Fatalf("the migrated file keeps a deprecated field: %+v\n%s", warning, report.Migrated)
		}
	}
	if !strings.Contains(report.Migrated, "pii: ") || strings.Contains(report.Migrated, "model_id") || strings.Contains(report.Migrated, "base_url") {
		t.Fatalf("the migrated file is not in the canonical form:\n%s", report.Migrated)
	}
}
