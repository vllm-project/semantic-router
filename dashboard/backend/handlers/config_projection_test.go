package handlers

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	routerprojection "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerprojection"
)

func TestPersistActiveConfigProjectionWritesReadModel(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	dslDir := filepath.Join(tempDir, ".vllm-sr")
	if err := os.MkdirAll(dslDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%s): %v", dslDir, err)
	}
	if err := os.WriteFile(filepath.Join(dslDir, "config.dsl"), []byte("ROUTE default-business { }"), 0o644); err != nil {
		t.Fatalf("WriteFile(config.dsl): %v", err)
	}

	if err := persistActiveConfigProjection(configPath, tempDir); err != nil {
		t.Fatalf("persistActiveConfigProjection() error = %v", err)
	}

	data, err := os.ReadFile(activeConfigProjectionPath(tempDir))
	if err != nil {
		t.Fatalf("ReadFile(projection): %v", err)
	}

	var projection routerprojection.ActiveConfigProjection
	if err := json.Unmarshal(data, &projection); err != nil {
		t.Fatalf("json.Unmarshal(projection): %v", err)
	}

	if !projection.Validation.Valid {
		t.Fatalf("expected valid projection, got %+v", projection.Validation)
	}
	if len(projection.Models) != 1 || projection.Models[0].Name != "test-model" {
		t.Fatalf("models = %+v, want test-model projection", projection.Models)
	}
	if len(projection.Decisions) != 1 || projection.Decisions[0].Name != "default-business" {
		t.Fatalf("decisions = %+v, want default-business projection", projection.Decisions)
	}
	if projection.DSLSnapshot == nil || projection.DSLSnapshot.Source != "archived_source" {
		t.Fatalf("dsl_snapshot = %+v, want archived_source snapshot", projection.DSLSnapshot)
	}
}

func TestApplyWrittenConfigPersistsActiveConfigProjection(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	previousData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile(configPath): %v", err)
	}

	updatedYAML := []byte(`
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    default_model: projection-model
    reasoning_families:
      qwen3:
        type: reasoning_effort
        parameter: reasoning_effort
  models:
    - name: projection-model
      reasoning_family: qwen3
      provider_model_id: projection-model
      backend_refs:
        - name: endpoint1
          endpoint: 127.0.0.1:8000
          protocol: http
          weight: 1
routing:
  modelCards:
    - name: projection-model
  decisions:
    - name: projection-route
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: projection-model
  signals:
    domains:
      - name: business
        description: Business requests
`)
	if writeErr := os.WriteFile(configPath, updatedYAML, 0o644); writeErr != nil {
		t.Fatalf("WriteFile(updated config): %v", writeErr)
	}

	if applyErr := applyWrittenConfig(configPath, tempDir, previousData, true); applyErr != nil {
		t.Fatalf("applyWrittenConfig() error = %v", applyErr)
	}

	data, err := os.ReadFile(activeConfigProjectionPath(tempDir))
	if err != nil {
		t.Fatalf("ReadFile(projection): %v", err)
	}

	var projection routerprojection.ActiveConfigProjection
	if err := json.Unmarshal(data, &projection); err != nil {
		t.Fatalf("json.Unmarshal(projection): %v", err)
	}

	if len(projection.Models) != 1 || projection.Models[0].Name != "projection-model" {
		t.Fatalf("models = %+v, want projection-model", projection.Models)
	}
	if len(projection.Decisions) != 1 || projection.Decisions[0].Name != "projection-route" {
		t.Fatalf("decisions = %+v, want projection-route", projection.Decisions)
	}
}
