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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
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
	return runValidationAs(t, document, config.GatewayStandalone)
}

func runValidationAs(t *testing.T, document string, gateway config.GatewayMode) (int, configValidationReport) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	status := validateConfigFile(path, gateway, false, &out)
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

// `vllm-sr serve` validates with the Router image, which does not mount the
// stack's state directory, so a key file the Dashboard saved is not there.
func TestValidateConfigLeavesAKeyFileToTheServingRouter(t *testing.T) {
	document := strings.Replace(outputValidationConfig, "        - endpoint: 127.0.0.1:8000\n",
		"        - endpoint: 127.0.0.1:8000\n          auth: {api_key_file: /app/.vllm-sr/provider-keys/key-1}\n", 1)
	document = strings.NewReplacer("CONDITION", "wants_text", "OUTPUT", "").Replace(document)
	if status, report := runValidation(t, document); status != 0 || !report.Valid {
		t.Fatalf("status %d, report %+v: a key file that is not mounted here must not fail validation", status, report)
	}
}

// httpsBackendConfig is a loadable document whose one backend is the HTTPS
// endpoint, with a private CA and server name, and the global block.
func httpsBackendConfig(endpoint, global string) string {
	backend := "        - name: local\n" +
		"          endpoint: " + endpoint + "\n" +
		"          provider: vllm\n" +
		"          tls:\n" +
		"            ca_file: /etc/vllm-sr/local-ca.pem\n" +
		"            server_name: tls-backend.test\n"
	document := strings.Replace(outputValidationConfig, "        - endpoint: 127.0.0.1:8000\n", backend, 1)
	return strings.NewReplacer("CONDITION", "wants_text", "OUTPUT", "").Replace(document) + global
}

// -validate-config refuses an HTTPS backend IP only where startup does: a
// standalone Router that routes. An ext_proc Router only logs it, and a
// Router with routing disabled builds no upstream.
func TestValidateConfigRefusesAnHTTPSBackendIPWhereStartupDoes(t *testing.T) {
	const refused = "providers.models[text-model].backend_refs[0] HTTPS endpoint must use a DNS hostname"
	routingDisabled := "global:\n  router:\n    enabled: false\n"
	cases := []struct {
		name     string
		gateway  config.GatewayMode
		endpoint string
		global   string
		refuse   bool
	}{
		{"standalone IPv4", config.GatewayStandalone, "https://127.0.0.1:8000/v1", "", true},
		{"standalone IPv6", config.GatewayStandalone, "https://[::1]:8000/v1", "", true},
		{"standalone hostname", config.GatewayStandalone, "https://localhost:8000/v1", "", false},
		{"ext_proc IPv4", config.GatewayExtProc, "https://127.0.0.1:8000/v1", "", false},
		{"standalone IPv4 without routing", config.GatewayStandalone, "https://127.0.0.1:8000/v1", routingDisabled, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			status, report := runValidationAs(t, httpsBackendConfig(tc.endpoint, tc.global), tc.gateway)
			if !tc.refuse {
				if status != 0 || !report.Valid {
					t.Fatalf("status %d, report %+v: the Router starts with this document", status, report)
				}
				return
			}
			if status != 1 || report.Valid {
				t.Fatalf("status %d, report %+v: the Router refuses this document at startup", status, report)
			}
			if !strings.Contains(report.Error, refused) {
				t.Fatalf("error %q is not the startup error %q", report.Error, refused)
			}
		})
	}
}

// referenceConfig is a loadable document with the listener timeout, when it
// is not empty, and one backend at endpoint; either may hold an environment
// reference.
func referenceConfig(timeout, endpoint string) string {
	document := outputValidationConfig
	if timeout != "" {
		document = strings.Replace(document, "    port: 8899\n", "    port: 8899\n    timeout: \""+timeout+"\"\n", 1)
	}
	backend := "        - name: local\n" +
		"          endpoint: \"" + endpoint + "\"\n" +
		"          provider: vllm\n"
	document = strings.Replace(document, "        - endpoint: 127.0.0.1:8000\n", backend, 1)
	return strings.NewReplacer("CONDITION", "wants_text", "OUTPUT", "").Replace(document)
}

// startupUpstreamError is the error a standalone Router that routes gets
// when it loads the document in environment and compiles its upstream.
func startupUpstreamError(t *testing.T, document string, environment map[string]string) error {
	t.Helper()
	for name, value := range environment {
		t.Setenv(name, value)
	}
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		return err
	}
	_, err = upstream.Compile(cfg)
	return err
}

// -validate-config keeps environment references as written, and the serving
// Router resolves them before it compiles the upstream, so standalone
// validation leaves a value that holds one to that Router. Every other value
// is checked as startup checks it, and the verdict is the one startup reaches
// in the deployment's environment.
func TestValidateConfigLeavesEnvironmentReferencesToTheServingRouter(t *testing.T) {
	environment := map[string]string{
		"LISTENER_TIMEOUT": "30s",
		"AZURE_RESOURCE":   "example",
		"AZURE_DEPLOYMENT": "gpt",
	}
	const plainBackend = "http://localhost:8000/v1"
	cases := []struct {
		name     string
		timeout  string
		endpoint string
		refused  string
	}{
		{"listener timeout reference", "${LISTENER_TIMEOUT}", plainBackend, ""},
		{"listener timeout reference with a default", "${LISTENER_TIMEOUT:-30s}", plainBackend, ""},
		{"unbraced listener timeout reference", "$LISTENER_TIMEOUT", plainBackend, ""},
		{"backend host reference", "", "https://${AZURE_RESOURCE}.openai.azure.com/openai/deployments/${AZURE_DEPLOYMENT}", ""},
		{"listener timeout that is not a duration", "soon", plainBackend, `listeners[http-8899].timeout "soon" is not a valid duration`},
		{"backend host placeholder no environment resolves", "", "https://{{resource}}.openai.azure.com/v1", "invalid base URL"},
		{"HTTPS backend IP beside a listener timeout reference", "${LISTENER_TIMEOUT}", "https://127.0.0.1:8000/v1", "HTTPS endpoint must use a DNS hostname"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			document := referenceConfig(tc.timeout, tc.endpoint)
			status, report := runValidation(t, document)
			startupErr := startupUpstreamError(t, document, environment)
			if tc.refused == "" {
				if startupErr != nil {
					t.Fatalf("startup refuses the document: %v", startupErr)
				}
				if status != 0 || !report.Valid {
					t.Fatalf("status %d, report %+v: the Router starts with this document", status, report)
				}
				return
			}
			if startupErr == nil || !strings.Contains(startupErr.Error(), tc.refused) {
				t.Fatalf("startup error %v, want one containing %q", startupErr, tc.refused)
			}
			if status != 1 || report.Valid || !strings.Contains(report.Error, tc.refused) {
				t.Fatalf("status %d, report %+v: want the startup error containing %q", status, report, tc.refused)
			}
		})
	}
}
