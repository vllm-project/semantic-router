/*
Copyright 2026 vLLM Semantic Router Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	"encoding/json"
	"strings"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func candidateCount(candidates *ComplexityCandidates) int {
	if candidates == nil {
		return 0
	}
	return len(candidates.Candidates)
}

func TestComplexityRulesConfig(t *testing.T) {
	tests := []struct {
		name   string
		config ComplexityRulesConfig
		valid  bool
	}{
		{
			name: "valid simple complexity rule",
			config: ComplexityRulesConfig{
				Name:        "code-complexity",
				Description: "Classify coding tasks by complexity",
				Threshold:   "0.7",
				Hard: &ComplexityCandidates{
					Candidates: []string{
						"Implement a distributed lock manager",
						"Design a database migration system",
					},
				},
				Easy: &ComplexityCandidates{
					Candidates: []string{
						"Write a function to reverse a string",
						"Create a class to represent a rectangle",
					},
				},
			},
			valid: true,
		},
		{
			name: "complexity rule with composer",
			config: ComplexityRulesConfig{
				Name:      "medical-complexity",
				Threshold: "0.7",
				Hard: &ComplexityCandidates{
					Candidates: []string{
						"Differential diagnosis for chest pain",
					},
				},
				Easy: &ComplexityCandidates{
					Candidates: []string{
						"What is normal body temperature?",
					},
				},
				Composer: &RuleComposition{
					Operator: "AND",
					Conditions: []CompositionCondition{
						{
							Type: "domain",
							Name: "medical",
						},
					},
				},
			},
			valid: true,
		},
		{
			name: "complexity rule with OR composer",
			config: ComplexityRulesConfig{
				Name:      "language-complexity",
				Threshold: "0.65",
				Hard: &ComplexityCandidates{
					Candidates: []string{"complex query"},
				},
				Easy: &ComplexityCandidates{
					Candidates: []string{"simple query"},
				},
				Composer: &RuleComposition{
					Operator: "OR",
					Conditions: []CompositionCondition{
						{
							Type: "language",
							Name: "french",
						},
						{
							Type: "language",
							Name: "german",
						},
					},
				},
			},
			valid: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Verify JSON marshaling/unmarshaling
			data, err := json.Marshal(tt.config)
			if err != nil {
				t.Errorf("json.Marshal() error = %v", err)
				return
			}

			var config ComplexityRulesConfig
			if err := json.Unmarshal(data, &config); err != nil {
				t.Errorf("json.Unmarshal() error = %v", err)
				return
			}

			// Verify required fields
			if config.Name != tt.config.Name {
				t.Errorf("Name = %v, want %v", config.Name, tt.config.Name)
			}
			if candidateCount(config.Hard) != candidateCount(tt.config.Hard) {
				t.Errorf("Hard.Candidates length = %v, want %v", candidateCount(config.Hard), candidateCount(tt.config.Hard))
			}
			if candidateCount(config.Easy) != candidateCount(tt.config.Easy) {
				t.Errorf("Easy.Candidates length = %v, want %v", candidateCount(config.Easy), candidateCount(tt.config.Easy))
			}

			// Verify composer if present
			if tt.config.Composer != nil {
				if config.Composer == nil {
					t.Error("Composer should not be nil")
					return
				}
				if config.Composer.Operator != tt.config.Composer.Operator {
					t.Errorf("Composer.Operator = %v, want %v", config.Composer.Operator, tt.config.Composer.Operator)
				}
				if len(config.Composer.Conditions) != len(tt.config.Composer.Conditions) {
					t.Errorf("Composer.Conditions length = %v, want %v", len(config.Composer.Conditions), len(tt.config.Composer.Conditions))
				}
			}
		})
	}
}

func TestConfigSpecWithNewFields(t *testing.T) {
	tests := []struct {
		name   string
		config ConfigSpec
	}{
		{
			name: "config with a response cache embedding",
			config: ConfigSpec{
				ResponseCache: &SemanticCacheConfig{
					Enabled:     true,
					BackendType: "memory",
					Embedding:   &EmbeddingBinding{Deployment: "qwen3-embedding", Dimension: 1024},
				},
			},
		},
		{
			name: "config with complexity_rules",
			config: ConfigSpec{
				ComplexityRules: []ComplexityRulesConfig{
					{
						Name:      "code-complexity",
						Threshold: "0.7",
						Hard: &ComplexityCandidates{
							Candidates: []string{"complex task"},
						},
						Easy: &ComplexityCandidates{
							Candidates: []string{"simple task"},
						},
					},
				},
			},
		},
		{
			name: "config with both a response cache and complexity_rules",
			config: ConfigSpec{
				ResponseCache: &SemanticCacheConfig{
					Enabled:     true,
					BackendType: "memory",
				},
				ComplexityRules: []ComplexityRulesConfig{
					{
						Name:      "code-complexity",
						Threshold: "0.7",
						Hard: &ComplexityCandidates{
							Candidates: []string{"Implement distributed system"},
						},
						Easy: &ComplexityCandidates{
							Candidates: []string{"Write hello world"},
						},
					},
					{
						Name:      "reasoning-complexity",
						Threshold: "0.65",
						Hard: &ComplexityCandidates{
							Candidates: []string{"Analyze geopolitical implications"},
						},
						Easy: &ComplexityCandidates{
							Candidates: []string{"What is the capital?"},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test that ConfigSpec can be marshaled
			data, err := json.Marshal(tt.config)
			if err != nil {
				t.Errorf("json.Marshal() error = %v", err)
				return
			}

			// Test that ConfigSpec can be unmarshaled
			var config ConfigSpec
			if err := json.Unmarshal(data, &config); err != nil {
				t.Errorf("json.Unmarshal() error = %v", err)
				return
			}

			// Verify the response cache's embedding if present
			if tt.config.ResponseCache != nil && tt.config.ResponseCache.Embedding != nil {
				if config.ResponseCache == nil || config.ResponseCache.Embedding == nil || *config.ResponseCache.Embedding != *tt.config.ResponseCache.Embedding {
					t.Errorf("ResponseCache.Embedding = %+v, want %+v", config.ResponseCache, tt.config.ResponseCache.Embedding)
					return
				}
			}

			// Verify complexity_rules if present
			if tt.config.ComplexityRules != nil {
				if len(config.ComplexityRules) != len(tt.config.ComplexityRules) {
					t.Errorf("ComplexityRules length = %v, want %v", len(config.ComplexityRules), len(tt.config.ComplexityRules))
				}
			}
		})
	}
}

func TestConfigSpecPreservesCanonicalRoutingAndAlgorithmObjects(t *testing.T) {
	config := ConfigSpec{
		Routing: &apiextensionsv1.JSON{Raw: []byte(`{
			"signals": {
				"keywords": [{"name": "critical_payment_event", "operator": "AND", "keywords": ["payment_failed", "critical"]}]
			}
		}`)},
		Decisions: []DecisionConfig{
			{
				Name: "agentic-session-route",
				Rules: RuleCombinationConfig{
					Operator: "AND",
					Conditions: []RuleConditionConfig{
						{Type: "keyword", Name: "critical_payment_event"},
					},
				},
				Algorithm: &apiextensionsv1.JSON{Raw: []byte(`{
					"type": "multi_factor",
					"multi_factor": {"weights": {"quality": 0.6, "cost": 0.4}}
				}`)},
			},
		},
	}

	data, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	serialized := string(data)
	if !strings.Contains(serialized, `"routing"`) ||
		!strings.Contains(serialized, `"keywords"`) ||
		!strings.Contains(serialized, `"multi_factor"`) {
		t.Fatalf("expected canonical routing and algorithm objects to serialize as public fields, got %s", serialized)
	}

	var decoded ConfigSpec
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if decoded.Routing == nil || !strings.Contains(string(decoded.Routing.Raw), "critical_payment_event") {
		t.Fatalf("expected routing raw object to round-trip, got %#v", decoded.Routing)
	}
	if len(decoded.Decisions) != 1 ||
		decoded.Decisions[0].Algorithm == nil ||
		!strings.Contains(string(decoded.Decisions[0].Algorithm.Raw), "multi_factor") {
		t.Fatalf("expected decision algorithm raw object to round-trip, got %#v", decoded.Decisions)
	}
}

func TestSemanticRouterWithNewFeatures(t *testing.T) {
	tests := []struct {
		name string
		sr   *SemanticRouter
	}{
		{
			name: "semantic router with a response cache",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-cache",
					Namespace: "default",
				},
				Spec: SemanticRouterSpec{
					Replicas: func() *int32 { i := int32(1); return &i }(),
					Config: ConfigSpec{
						ResponseCache: &SemanticCacheConfig{
							Enabled:     true,
							BackendType: "memory",
						},
					},
				},
			},
		},
		{
			name: "semantic router with complexity routing",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-complexity",
					Namespace: "default",
				},
				Spec: SemanticRouterSpec{
					Replicas: func() *int32 { i := int32(1); return &i }(),
					Config: ConfigSpec{
						ComplexityRules: []ComplexityRulesConfig{
							{
								Name:      "code-complexity",
								Threshold: "0.7",
								Hard: &ComplexityCandidates{
									Candidates: []string{
										"Implement a distributed lock manager",
										"Design a database migration system",
									},
								},
								Easy: &ComplexityCandidates{
									Candidates: []string{
										"Write a function to reverse a string",
										"Create a simple counter",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test JSON marshaling of full SemanticRouter object
			data, err := json.Marshal(tt.sr)
			if err != nil {
				t.Errorf("json.Marshal() error = %v", err)
				return
			}

			// Test JSON unmarshaling
			var sr SemanticRouter
			if err := json.Unmarshal(data, &sr); err != nil {
				t.Errorf("json.Unmarshal() error = %v", err)
				return
			}

			// Verify metadata
			if sr.Name != tt.sr.Name {
				t.Errorf("Name = %v, want %v", sr.Name, tt.sr.Name)
			}
			if sr.Namespace != tt.sr.Namespace {
				t.Errorf("Namespace = %v, want %v", sr.Namespace, tt.sr.Namespace)
			}

			// Verify spec fields are preserved
			if *sr.Spec.Replicas != *tt.sr.Spec.Replicas {
				t.Errorf("Replicas = %v, want %v", *sr.Spec.Replicas, *tt.sr.Spec.Replicas)
			}
		})
	}
}
