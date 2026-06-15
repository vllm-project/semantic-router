package config

import (
	"strings"
	"testing"
)

func TestValidateConfigStructureRejectsRemovedStructureSpanDistance(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				StructureRules: []StructureRule{
					{
						Name: "tight_first_then_gap",
						Feature: StructureFeature{
							Type: "span_distance",
							Source: StructureSource{
								Type: "marker_pair",
							},
						},
					},
				},
			},
		},
	}

	err := validateConfigStructure(cfg)
	if err == nil {
		t.Fatal("expected validateConfigStructure to reject span_distance")
	}
	if !strings.Contains(err.Error(), `unsupported feature.type "span_distance"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateConfigStructureRejectsRemovedStructureMarkerPair(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				StructureRules: []StructureRule{
					{
						Name: "tight_first_then_gap",
						Feature: StructureFeature{
							Type: "count",
							Source: StructureSource{
								Type: "marker_pair",
							},
						},
					},
				},
			},
		},
	}

	err := validateConfigStructure(cfg)
	if err == nil {
		t.Fatal("expected validateConfigStructure to reject marker_pair")
	}
	if !strings.Contains(err.Error(), `unsupported feature.source.type "marker_pair"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateConfigStructureAcceptsDensityWithoutNormalizeBy(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				StructureRules: []StructureRule{
					{
						Name: "constraint_dense",
						Feature: StructureFeature{
							Type: "density",
							Source: StructureSource{
								Type:     "keyword_set",
								Keywords: []string{"at most", "不超过"},
							},
						},
					},
				},
			},
		},
	}

	if err := validateConfigStructure(cfg); err != nil {
		t.Fatalf("expected density rule without normalize_by to validate, got %v", err)
	}
}

func TestParseYAMLBytesRejectsRemovedStructureNormalizeBy(t *testing.T) {
	yamlContent := []byte(`
version: v0.3
listeners:
  - name: default
    address: 0.0.0.0
    port: 8080
providers:
  defaults: {}
  models: []
routing:
  modelCards: []
  signals:
    keywords: []
    embeddings: []
    domains: []
    fact_check: []
    user_feedbacks: []
    preferences: []
    language: []
    context: []
    structure:
      - name: constraint_dense
        feature:
          type: density
          normalize_by: char_count
          source:
            type: keyword_set
            keywords: ["最多"]
    complexity: []
    modality: []
    role_bindings: []
    jailbreak: []
    pii: []
  projections:
    partitions: []
    scores: []
    mappings: []
  decisions: []
global:
  router: {}
  services: {}
  stores: {}
  integrations: {}
  model_catalog: {}
`)

	_, err := ParseYAMLBytes(yamlContent)
	if err == nil {
		t.Fatal("expected ParseYAMLBytes to reject removed structure normalize_by")
	}
	if !strings.Contains(err.Error(), "feature.normalize_by") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestParseRoutingYAMLBytesRejectsRemovedStructureNormalizeBy(t *testing.T) {
	yamlContent := []byte(`
routing:
  signals:
    structure:
      - name: constraint_dense
        feature:
          type: density
          normalize_by: char_count
          source:
            type: keyword_set
            keywords: ["最多"]
  decisions: []
`)

	_, err := ParseRoutingYAMLBytes(yamlContent)
	if err == nil {
		t.Fatal("expected ParseRoutingYAMLBytes to reject removed structure normalize_by")
	}
	if !strings.Contains(err.Error(), "feature.normalize_by") {
		t.Fatalf("unexpected error: %v", err)
	}
}
