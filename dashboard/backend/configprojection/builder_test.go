package configprojection

import (
	"encoding/json"
	"strings"
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
    model: test-model
  models:
    - name: test-model
      reasoning:
        family: qwen3
      provider_model_id: test-model
      backend_refs:
        - name: endpoint1
          provider: vllm
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

func TestBuildSnapshotDecisionRuleLimits(t *testing.T) {
	data := testCanonicalYAML + "\nglobal:\n  router:\n    decision_rule_limits:\n      max_depth: 32\n      max_nodes: 512\n"
	snapshot, err := BuildSnapshot(RefreshInput{YAMLBytes: []byte(data)})
	if err != nil {
		t.Fatal(err)
	}
	var plugins pluginsProjection
	if err := json.Unmarshal(snapshot.Plugins, &plugins); err != nil {
		t.Fatal(err)
	}
	depth, nodes, err := plugins.Global.Router.DecisionRuleLimits.Effective()
	if err != nil || depth != 32 || nodes != 512 {
		t.Fatalf("projection lost custom limits: %d/%d, %v", depth, nodes, err)
	}
	data = strings.Replace(data, "max_nodes: 512", "max_nodes: 1", 1)
	if _, err := BuildSnapshot(RefreshInput{YAMLBytes: []byte(data)}); err == nil || !strings.Contains(err.Error(), "node count 2 exceeds max_nodes=1") {
		t.Fatalf("projection accepted oversized candidate: %v", err)
	}
}

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

	if !strings.Contains(string(snapshot.Signals), "business") {
		t.Fatalf("expected business domain in signals JSON, got %s", snapshot.Signals)
	}

	if !strings.Contains(string(snapshot.Decisions), "default-business") {
		t.Fatalf("expected default-business decision in JSON, got %s", snapshot.Decisions)
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
