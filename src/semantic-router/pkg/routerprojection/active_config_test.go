package routerprojection

import (
	"strings"
	"testing"
)

func TestBuildActiveConfigProjectionIncludesRoutingReadModel(t *testing.T) {
	configYAML := []byte(`
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    default_model: general
    reasoning_families:
      qwen3:
        type: reasoning_effort
        parameter: reasoning_effort
  models:
    - name: general
      reasoning_family: qwen3
      provider_model_id: general-model
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:8000
          protocol: http
          weight: 1
routing:
  modelCards:
    - name: general
      capabilities: ["chat", "tool_use"]
  signals:
    domains:
      - name: business
        description: Business requests
    keywords:
      - name: invoice_keywords
        operator: OR
        keywords: ["invoice"]
  decisions:
    - name: business-default
      priority: 10
      rules:
        operator: OR
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: general
      plugins:
        - type: router_replay
          configuration:
            enabled: true
`)

	projection, err := BuildActiveConfigProjection(configYAML, []byte("ROUTE business-default { }"))
	if err != nil {
		t.Fatalf("BuildActiveConfigProjection() error = %v", err)
	}

	if !projection.Validation.Valid {
		t.Fatalf("expected valid projection, got %+v", projection.Validation)
	}
	if projection.SchemaVersion != projectionSchemaVersion {
		t.Fatalf("schema_version = %q, want %q", projection.SchemaVersion, projectionSchemaVersion)
	}
	if len(projection.Models) != 1 || projection.Models[0].Name != "general" {
		t.Fatalf("models = %+v, want one general model", projection.Models)
	}
	if got := projection.Models[0].Capabilities; len(got) != 2 || got[0] != "chat" {
		t.Fatalf("model capabilities = %+v, want chat/tool_use", got)
	}
	if len(projection.Signals) != 2 {
		t.Fatalf("signals = %+v, want 2 named signals", projection.Signals)
	}
	if len(projection.Decisions) != 1 || projection.Decisions[0].Name != "business-default" {
		t.Fatalf("decisions = %+v, want business-default", projection.Decisions)
	}
	if len(projection.Plugins) != 1 || projection.Plugins[0].Type != "router_replay" {
		t.Fatalf("plugins = %+v, want router_replay", projection.Plugins)
	}
	if projection.DSLSnapshot == nil || projection.DSLSnapshot.Source != "archived_source" {
		t.Fatalf("dsl_snapshot = %+v, want archived_source snapshot", projection.DSLSnapshot)
	}
	if !strings.Contains(projection.DSLSnapshot.Text, "business-default") {
		t.Fatalf("dsl_snapshot.text = %q, want archived DSL text", projection.DSLSnapshot.Text)
	}
}

func TestBuildActiveConfigProjectionFallsBackToDecompiledRouting(t *testing.T) {
	configYAML := []byte(`
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    default_model: general
    reasoning_families:
      qwen3:
        type: reasoning_effort
        parameter: reasoning_effort
  models:
    - name: general
      reasoning_family: qwen3
      provider_model_id: general-model
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:8000
          protocol: http
          weight: 1
routing:
  modelCards:
    - name: general
  decisions:
    - name: default-route
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: keyword
            name: fallback
      modelRefs:
        - model: general
`)

	projection, err := BuildActiveConfigProjection(configYAML, nil)
	if err != nil {
		t.Fatalf("BuildActiveConfigProjection() error = %v", err)
	}
	if projection.DSLSnapshot == nil || projection.DSLSnapshot.Source != "decompiled_routing" {
		t.Fatalf("dsl_snapshot = %+v, want decompiled_routing snapshot", projection.DSLSnapshot)
	}
	if !strings.Contains(projection.DSLSnapshot.Text, "ROUTE default-route") {
		t.Fatalf("dsl_snapshot.text = %q, want decompiled route", projection.DSLSnapshot.Text)
	}
}
