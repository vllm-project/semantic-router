package configprojection

import (
	"encoding/json"
	"testing"
)

const testCanonicalYAML = `
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    default_model: test-model
    reasoning_families:
      qwen3:
        type: reasoning_effort
        parameter: reasoning_effort
  models:
    - name: test-model
      reasoning_family: qwen3
      provider_model_id: test-model
      backend_refs:
        - name: endpoint1
          endpoint: 127.0.0.1:8000
          protocol: http
          weight: 1
routing:
  modelCards:
    - name: test-model
  signals:
    domains:
      - name: business
        description: Business and management related queries
  decisions:
    - name: default-business
      description: Route business requests to the default model
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
          use_reasoning: false
`

func TestBuildSnapshotExtractsEntities(t *testing.T) {
	t.Parallel()

	snapshot, err := BuildSnapshot(RefreshInput{
		Version:     "20260101-120000",
		Source:      SourceDSL,
		YAMLBytes:   []byte(testCanonicalYAML),
		DSLSnapshot: "ROUTE default-business",
	})
	if err != nil {
		t.Fatalf("BuildSnapshot: %v", err)
	}

	if snapshot.Validation.Status != "ok" {
		t.Fatalf("expected validation ok, got %+v", snapshot.Validation)
	}
	if snapshot.DSLSnapshot != "ROUTE default-business" {
		t.Fatalf("unexpected dsl snapshot: %q", snapshot.DSLSnapshot)
	}
	if snapshot.YAMLHash == "" {
		t.Fatal("expected yaml hash")
	}

	var models modelsProjection
	if err := json.Unmarshal(snapshot.Models, &models); err != nil {
		t.Fatalf("unmarshal models: %v", err)
	}
	if len(models.ProviderModels) != 1 || models.ProviderModels[0].Name != "test-model" {
		t.Fatalf("unexpected provider models: %+v", models.ProviderModels)
	}
	if len(models.ModelCards) != 1 || models.ModelCards[0].Name != "test-model" {
		t.Fatalf("unexpected model cards: %+v", models.ModelCards)
	}

	var signals struct {
		Domains []struct {
			Name string `json:"Name"`
		} `json:"Domains"`
	}
	if err := json.Unmarshal(snapshot.Signals, &signals); err != nil {
		t.Fatalf("unmarshal signals: %v", err)
	}
	if len(signals.Domains) != 1 || signals.Domains[0].Name != "business" {
		t.Fatalf("unexpected signals: %+v", signals)
	}

	var decisions []struct {
		Name string `json:"Name"`
	}
	if err := json.Unmarshal(snapshot.Decisions, &decisions); err != nil {
		t.Fatalf("unmarshal decisions: %v", err)
	}
	if len(decisions) != 1 || decisions[0].Name != "default-business" {
		t.Fatalf("unexpected decisions: %+v", decisions)
	}
}

func TestBuildSnapshotRejectsInvalidYAML(t *testing.T) {
	t.Parallel()

	_, err := BuildSnapshot(RefreshInput{
		YAMLBytes: []byte("routing: ["),
	})
	if err == nil {
		t.Fatal("expected invalid YAML to fail")
	}
}
