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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestEmbeddingModelsConfig(t *testing.T) {
	tests := []struct {
		name   string
		config EmbeddingModelsConfig
		want   string // Expected JSON output
	}{
		{
			name: "qwen3 model configuration",
			config: EmbeddingModelsConfig{
				Qwen3ModelPath: "models/qwen3-embedding",
				UseCPU:         true,
			},
			want: `{"qwen3_model_path":"models/qwen3-embedding","use_cpu":true}`,
		},
		{
			name: "gemma model configuration",
			config: EmbeddingModelsConfig{
				GemmaModelPath: "models/gemma-embedding",
				UseCPU:         true,
			},
			want: `{"gemma_model_path":"models/gemma-embedding","use_cpu":true}`,
		},
		{
			name: "mmbert model configuration",
			config: EmbeddingModelsConfig{
				MmBertModelPath: "models/mmbert-embedding",
				UseCPU:          true,
				HNSWConfig: &HNSWEmbeddingConfig{
					ModelType:       "mmbert",
					TargetLayer:     6,
					TargetDimension: 256,
				},
			},
			want: `{"mmbert_model_path":"models/mmbert-embedding","use_cpu":true,"hnsw_config":{"model_type":"mmbert","target_dimension":256,"target_layer":6}}`,
		},
		{
			name: "all models configured",
			config: EmbeddingModelsConfig{
				Qwen3ModelPath:  "models/qwen3-embedding",
				GemmaModelPath:  "models/gemma-embedding",
				MmBertModelPath: "models/mmbert-embedding",
				UseCPU:          true,
			},
			want: `{"qwen3_model_path":"models/qwen3-embedding","gemma_model_path":"models/gemma-embedding","mmbert_model_path":"models/mmbert-embedding","use_cpu":true}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.config)
			if err != nil {
				t.Errorf("json.Marshal() error = %v", err)
				return
			}
			got := string(data)
			if got != tt.want {
				t.Errorf("json.Marshal() = %v, want %v", got, tt.want)
			}

			// Test unmarshaling
			var config EmbeddingModelsConfig
			if err := json.Unmarshal(data, &config); err != nil {
				t.Errorf("json.Unmarshal() error = %v", err)
			}
		})
	}
}

func TestHNSWEmbeddingConfig(t *testing.T) {
	tests := []struct {
		name   string
		config HNSWEmbeddingConfig
		valid  bool
	}{
		{
			name: "valid qwen3 configuration",
			config: HNSWEmbeddingConfig{
				ModelType:          "qwen3",
				PreloadEmbeddings:  true,
				TargetDimension:    768,
				EnableSoftMatching: true,
				MinScoreThreshold:  "0.5",
			},
			valid: true,
		},
		{
			name: "valid mmbert with layer 3 (fastest)",
			config: HNSWEmbeddingConfig{
				ModelType:       "mmbert",
				TargetLayer:     3,
				TargetDimension: 64,
			},
			valid: true,
		},
		{
			name: "valid mmbert with layer 6 (balanced)",
			config: HNSWEmbeddingConfig{
				ModelType:       "mmbert",
				TargetLayer:     6,
				TargetDimension: 256,
			},
			valid: true,
		},
		{
			name: "valid mmbert with layer 11 (high quality)",
			config: HNSWEmbeddingConfig{
				ModelType:       "mmbert",
				TargetLayer:     11,
				TargetDimension: 512,
			},
			valid: true,
		},
		{
			name: "valid mmbert with layer 22 (full)",
			config: HNSWEmbeddingConfig{
				ModelType:       "mmbert",
				TargetLayer:     22,
				TargetDimension: 768,
			},
			valid: true,
		},
		{
			name: "valid gemma configuration",
			config: HNSWEmbeddingConfig{
				ModelType:       "gemma",
				TargetDimension: 768,
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

			var config HNSWEmbeddingConfig
			if err := json.Unmarshal(data, &config); err != nil {
				t.Errorf("json.Unmarshal() error = %v", err)
			}

			// Verify field values are preserved
			if config.ModelType != tt.config.ModelType {
				t.Errorf("ModelType = %v, want %v", config.ModelType, tt.config.ModelType)
			}
			if config.TargetLayer != tt.config.TargetLayer {
				t.Errorf("TargetLayer = %v, want %v", config.TargetLayer, tt.config.TargetLayer)
			}
			if config.TargetDimension != tt.config.TargetDimension {
				t.Errorf("TargetDimension = %v, want %v", config.TargetDimension, tt.config.TargetDimension)
			}
		})
	}
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
				Hard: ComplexityCandidates{
					Candidates: []string{
						"Implement a distributed lock manager",
						"Design a database migration system",
					},
				},
				Easy: ComplexityCandidates{
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
				Hard: ComplexityCandidates{
					Candidates: []string{
						"Differential diagnosis for chest pain",
					},
				},
				Easy: ComplexityCandidates{
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
				Hard: ComplexityCandidates{
					Candidates: []string{"complex query"},
				},
				Easy: ComplexityCandidates{
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
			if len(config.Hard.Candidates) != len(tt.config.Hard.Candidates) {
				t.Errorf("Hard.Candidates length = %v, want %v", len(config.Hard.Candidates), len(tt.config.Hard.Candidates))
			}
			if len(config.Easy.Candidates) != len(tt.config.Easy.Candidates) {
				t.Errorf("Easy.Candidates length = %v, want %v", len(config.Easy.Candidates), len(tt.config.Easy.Candidates))
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

func TestSemanticCacheEmbeddingModel(t *testing.T) {
	tests := []struct {
		name  string
		model string
		valid bool
	}{
		{
			name:  "bert model (default)",
			model: "bert",
			valid: true,
		},
		{
			name:  "qwen3 model",
			model: "qwen3",
			valid: true,
		},
		{
			name:  "gemma model",
			model: "gemma",
			valid: true,
		},
		{
			name:  "mmbert model",
			model: "mmbert",
			valid: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := SemanticCacheConfig{
				Enabled:        true,
				BackendType:    "memory",
				EmbeddingModel: tt.model,
			}

			// Verify JSON marshaling includes the embedding_model
			data, err := json.Marshal(config)
			if err != nil {
				t.Errorf("json.Marshal() error = %v", err)
				return
			}

			var unmarshaled SemanticCacheConfig
			if err := json.Unmarshal(data, &unmarshaled); err != nil {
				t.Errorf("json.Unmarshal() error = %v", err)
				return
			}

			if unmarshaled.EmbeddingModel != tt.model {
				t.Errorf("EmbeddingModel = %v, want %v", unmarshaled.EmbeddingModel, tt.model)
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
			name: "config with embedding_models",
			config: ConfigSpec{
				EmbeddingModels: &EmbeddingModelsConfig{
					Qwen3ModelPath: "models/qwen3-embedding",
					UseCPU:         true,
				},
				SemanticCache: &SemanticCacheConfig{
					Enabled:        true,
					BackendType:    "memory",
					EmbeddingModel: "qwen3",
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
						Hard: ComplexityCandidates{
							Candidates: []string{"complex task"},
						},
						Easy: ComplexityCandidates{
							Candidates: []string{"simple task"},
						},
					},
				},
			},
		},
		{
			name: "config with both embedding_models and complexity_rules",
			config: ConfigSpec{
				EmbeddingModels: &EmbeddingModelsConfig{
					MmBertModelPath: "models/mmbert-embedding",
					UseCPU:          true,
					HNSWConfig: &HNSWEmbeddingConfig{
						ModelType:       "mmbert",
						TargetLayer:     6,
						TargetDimension: 256,
					},
				},
				SemanticCache: &SemanticCacheConfig{
					Enabled:        true,
					BackendType:    "memory",
					EmbeddingModel: "mmbert",
				},
				ComplexityRules: []ComplexityRulesConfig{
					{
						Name:      "code-complexity",
						Threshold: "0.7",
						Hard: ComplexityCandidates{
							Candidates: []string{"Implement distributed system"},
						},
						Easy: ComplexityCandidates{
							Candidates: []string{"Write hello world"},
						},
					},
					{
						Name:      "reasoning-complexity",
						Threshold: "0.65",
						Hard: ComplexityCandidates{
							Candidates: []string{"Analyze geopolitical implications"},
						},
						Easy: ComplexityCandidates{
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

			// Verify embedding_models if present
			if tt.config.EmbeddingModels != nil {
				if config.EmbeddingModels == nil {
					t.Error("EmbeddingModels should not be nil")
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

func TestSemanticRouterWithNewFeatures(t *testing.T) {
	tests := []struct {
		name string
		sr   *SemanticRouter
	}{
		{
			name: "semantic router with mmbert embeddings",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-mmbert",
					Namespace: "default",
				},
				Spec: SemanticRouterSpec{
					Replicas: func() *int32 { i := int32(1); return &i }(),
					Config: ConfigSpec{
						EmbeddingModels: &EmbeddingModelsConfig{
							MmBertModelPath: "models/mmbert-embedding",
							UseCPU:          true,
							HNSWConfig: &HNSWEmbeddingConfig{
								ModelType:       "mmbert",
								TargetLayer:     6,
								TargetDimension: 256,
							},
						},
						SemanticCache: &SemanticCacheConfig{
							Enabled:        true,
							BackendType:    "memory",
							EmbeddingModel: "mmbert",
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
						EmbeddingModels: &EmbeddingModelsConfig{
							Qwen3ModelPath: "models/qwen3-embedding",
							UseCPU:         true,
						},
						ComplexityRules: []ComplexityRulesConfig{
							{
								Name:      "code-complexity",
								Threshold: "0.7",
								Hard: ComplexityCandidates{
									Candidates: []string{
										"Implement a distributed lock manager",
										"Design a database migration system",
									},
								},
								Easy: ComplexityCandidates{
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
