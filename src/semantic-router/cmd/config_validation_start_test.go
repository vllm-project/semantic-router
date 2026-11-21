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

const startParityConfig = `version: v0.5
LISTENERS
providers:
  DEFAULTS
  models:
    - name: m
      backend_refs:
        - endpoint: 127.0.0.1:8000
recipes:
  - name: default
GLOBAL
`

// -validate-config refuses what the Router refuses when it starts with the
// same file in the same gateway mode, and nothing more.
func TestValidateConfigRefusesWhatStartRefuses(t *testing.T) {
	listener := "listeners:\n  - {name: http, address: 127.0.0.1, port: 8899}"
	for _, tc := range []struct {
		name, listeners, defaults, global string
		mode                              config.GatewayMode
		want                              string
	}{
		{name: "a known store", listeners: listener, defaults: "defaults: {model: m}", global: "global:\n  services:\n    router_replay: {store_backend: memory}", mode: config.GatewayStandalone},
		{
			name: "an unknown store", listeners: listener, defaults: "defaults: {model: m}", global: "global:\n  services:\n    router_replay: {store_backend: bogus}", mode: config.GatewayStandalone,
			want: `global.services.router_replay.store_backend "bogus" is not supported: use one of file, memory, redis, postgres, milvus, qdrant`,
		},
		{
			name: "standalone without a listener", defaults: "defaults: {model: m}", mode: config.GatewayStandalone,
			want: "standalone mode needs at least one listener in the configuration",
		},
		{name: "extproc without a listener", defaults: "defaults: {model: m}", mode: config.GatewayExtProc},
	} {
		t.Run(tc.name, func(t *testing.T) {
			document := strings.NewReplacer("LISTENERS", tc.listeners, "DEFAULTS", tc.defaults, "GLOBAL", tc.global).Replace(startParityConfig)
			path := filepath.Join(t.TempDir(), "config.yaml")
			if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
				t.Fatal(err)
			}
			var out bytes.Buffer
			status := validateConfigFile(path, tc.mode, false, &out)
			var report configValidationReport
			if err := json.Unmarshal(out.Bytes(), &report); err != nil {
				t.Fatalf("stdout is not one JSON report: %q", out.String())
			}
			if tc.want == "" && (status != 0 || !report.Valid) {
				t.Fatalf("status %d, report %+v: the Router starts with this file", status, report)
			}
			if tc.want != "" && (status != 1 || report.Valid || !strings.Contains(report.Error, tc.want)) {
				t.Fatalf("status %d, report %+v: want the start's refusal %q", status, report, tc.want)
			}
		})
	}
}
