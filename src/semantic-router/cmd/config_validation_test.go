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

const outputValidationConfig = `
version: v0.5
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
recipes:
  - name: default
    signals:
      modality:
        - name: wants_text
          description: Text-only requests.
          labels: [text]
        - name: wants_image
          description: Requests for a generated image.
          labels: [image]
    decisions:
      - name: image-gen
        rules:
          operator: AND
          conditions:
            - type: modality
              name: CONDITION
        OUTPUT
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
	document := strings.NewReplacer("CONDITION", "wants_image", "OUTPUT", "output: {kind: image, action: generate}").Replace(outputValidationConfig)
	status, report := runValidation(t, document)
	if status != 1 || report.Valid {
		t.Fatalf("status %d, report %+v: an image decision whose model cannot generate images must not validate", status, report)
	}
	if !strings.Contains(report.Error, `output_capability_missing: model "text-model" does not declare image_generation in capabilities`) {
		t.Fatalf("error %q is not the Router's", report.Error)
	}
}

// A document that loads exits 0 and carries the warnings the Router logs.
func TestValidateConfigReportsTheRoutersWarnings(t *testing.T) {
	document := strings.NewReplacer(
		"CONDITION", "wants_text", "OUTPUT", "",
		"    signals:\n", "    signals:\n      decision:\n        - name: needs_tools\n          question: {type: noul, instructions: \"Does the request need a tool?\"}\n",
	).Replace(outputValidationConfig)
	status, report := runValidation(t, document)
	if status != 0 || !report.Valid || report.Error != "" {
		t.Fatalf("status %d, report %+v: the document loads", status, report)
	}
	if len(report.Warnings) != 1 || report.Warnings[0].Code != config.CodeRenamedSignalType {
		t.Fatalf("warnings = %+v, want %s", report.Warnings, config.CodeRenamedSignalType)
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

// -migrate-config writes a file the Router refuses for its removed model
// fields in the forms that replace them; the migrated file loads.
func TestMigrateConfigRewritesRemovedModelFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	document := `version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "http://127.0.0.1:8000/v1", provider: vllm, weight: 1}]
recipes:
  - name: default
    signals:
      pii: [{name: personal, predicate: {gte: 0.5}}]
    decisions:
      - name: route
        priority: 100
        rules: {operator: OR, conditions: [{type: pii, name: personal}]}
        modelRefs: [{model: general}]
global:
  model_catalog:
    bindings:
      pii: {deployment: primary, contract: decision.v1}
`
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if status := validateConfigFile(path, config.GatewayStandalone, true, &out); status != 1 {
		t.Fatalf("status %d for a file written in removed forms: %s", status, out.String())
	}
	var report configValidationReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Valid || !strings.Contains(report.Error, "removed_field") || report.Migrated == "" || report.MigrationError != "" {
		t.Fatalf("migration report %+v", report)
	}
	for _, note := range report.Notes {
		if note.Code != "migration_change" {
			t.Fatalf("a rewrite needs an action: %+v", note)
		}
	}
	migrated, err := config.ParseYAMLBytesWithoutEnvExpansion([]byte(report.Migrated))
	if err != nil {
		t.Fatalf("the migrated file does not load: %v\n%s", err, report.Migrated)
	}
	if warnings := config.Warnings(migrated); len(warnings) > 0 {
		t.Fatalf("the migrated file loads with warnings: %+v\n%s", warnings, report.Migrated)
	}
	if !strings.Contains(report.Migrated, "pii: primary") || strings.Contains(report.Migrated, "contract") || strings.Contains(report.Migrated, "base_url") {
		t.Fatalf("the migrated file is not in the canonical form:\n%s", report.Migrated)
	}
}
