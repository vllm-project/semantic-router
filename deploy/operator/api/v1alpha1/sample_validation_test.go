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
	"os"
	"path/filepath"
	"testing"

	"k8s.io/apimachinery/pkg/util/yaml"
)

func TestSampleCRValidation(t *testing.T) {
	// Get the project root directory
	projectRoot := filepath.Join("..", "..", "..")
	samplesDir := filepath.Join(projectRoot, "config", "samples")

	tests := []struct {
		name     string
		filename string
	}{
		{
			name:     "mmbert sample CR",
			filename: "vllm.ai_v1alpha1_semanticrouter_mmbert.yaml",
		},
		{
			name:     "complexity routing sample CR",
			filename: "vllm.ai_v1alpha1_semanticrouter_complexity.yaml",
		},
		{
			name:     "simple sample CR",
			filename: "vllm.ai_v1alpha1_semanticrouter_simple.yaml",
		},
		{
			name:     "redis cache sample CR",
			filename: "vllm.ai_v1alpha1_semanticrouter_redis_cache.yaml",
		},
		{
			name:     "milvus cache sample CR",
			filename: "vllm.ai_v1alpha1_semanticrouter_milvus_cache.yaml",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			samplePath := filepath.Join(samplesDir, tt.filename)

			// Read the sample YAML file
			data, err := os.ReadFile(samplePath)
			if err != nil {
				t.Skipf("Sample file not found: %s (this is expected during development)", samplePath)
				return
			}

			// Parse the YAML into a SemanticRouter object
			var sr SemanticRouter
			if err := yaml.Unmarshal(data, &sr); err != nil {
				t.Errorf("Failed to unmarshal sample CR: %v", err)
				return
			}

			// Validate the CR using webhook validation
			_, err = sr.ValidateCreate()
			if err != nil {
				t.Errorf("Sample CR validation failed: %v", err)
			}

			// Additional checks for specific samples
			if tt.filename == "vllm.ai_v1alpha1_semanticrouter_mmbert.yaml" {
				if sr.Spec.Config.EmbeddingModels == nil {
					t.Error("mmbert sample should have embedding_models configured")
				} else {
					if sr.Spec.Config.EmbeddingModels.MmBertModelPath == "" {
						t.Error("mmbert sample should have mmbert_model_path set")
					}
					if sr.Spec.Config.EmbeddingModels.HNSWConfig == nil {
						t.Error("mmbert sample should have hnsw_config")
					} else {
						if sr.Spec.Config.EmbeddingModels.HNSWConfig.ModelType != "mmbert" {
							t.Errorf("mmbert sample hnsw_config model_type = %v, want mmbert", sr.Spec.Config.EmbeddingModels.HNSWConfig.ModelType)
						}
						// Verify target_layer is one of the valid values
						validLayers := map[int]bool{3: true, 6: true, 11: true, 22: true}
						if !validLayers[sr.Spec.Config.EmbeddingModels.HNSWConfig.TargetLayer] {
							t.Errorf("mmbert sample target_layer = %v, want one of 3, 6, 11, 22", sr.Spec.Config.EmbeddingModels.HNSWConfig.TargetLayer)
						}
					}
				}
				if sr.Spec.Config.SemanticCache != nil {
					if sr.Spec.Config.SemanticCache.EmbeddingModel != "mmbert" {
						t.Errorf("mmbert sample should use embedding_model: mmbert, got %v", sr.Spec.Config.SemanticCache.EmbeddingModel)
					}
				}
			}

			if tt.filename == "vllm.ai_v1alpha1_semanticrouter_complexity.yaml" {
				if len(sr.Spec.Config.ComplexityRules) == 0 {
					t.Error("complexity sample should have complexity_rules configured")
				}
				// Verify structure of complexity rules
				for _, rule := range sr.Spec.Config.ComplexityRules {
					if rule.Name == "" {
						t.Error("complexity rule should have a name")
					}
					if len(rule.Hard.Candidates) == 0 {
						t.Errorf("complexity rule %s should have hard candidates", rule.Name)
					}
					if len(rule.Easy.Candidates) == 0 {
						t.Errorf("complexity rule %s should have easy candidates", rule.Name)
					}
				}
			}

			if tt.filename == "vllm.ai_v1alpha1_semanticrouter_redis_cache.yaml" {
				if sr.Spec.Config.EmbeddingModels == nil {
					t.Error("redis cache sample should have embedding_models configured")
				}
				if sr.Spec.Config.SemanticCache != nil {
					if sr.Spec.Config.SemanticCache.BackendType != "redis" {
						t.Errorf("redis cache sample backend_type = %v, want redis", sr.Spec.Config.SemanticCache.BackendType)
					}
					// Should use qwen3 or another embedding model
					if sr.Spec.Config.SemanticCache.EmbeddingModel != "" && sr.Spec.Config.SemanticCache.EmbeddingModel == "bert" {
						// This is fine, but ideally should showcase new embedding models
						t.Logf("redis cache sample could showcase new embedding models (qwen3/gemma)")
					}
				}
			}

			if tt.filename == "vllm.ai_v1alpha1_semanticrouter_milvus_cache.yaml" {
				if sr.Spec.Config.EmbeddingModels == nil {
					t.Error("milvus cache sample should have embedding_models configured")
				}
				if sr.Spec.Config.SemanticCache != nil {
					if sr.Spec.Config.SemanticCache.BackendType != "milvus" {
						t.Errorf("milvus cache sample backend_type = %v, want milvus", sr.Spec.Config.SemanticCache.BackendType)
					}
				}
			}
		})
	}
}

func TestSampleCRsParseable(t *testing.T) {
	// Test that all sample CRs can be parsed without errors
	projectRoot := filepath.Join("..", "..", "..")
	samplesDir := filepath.Join(projectRoot, "config", "samples")

	entries, err := os.ReadDir(samplesDir)
	if err != nil {
		t.Skipf("Samples directory not found: %s", samplesDir)
		return
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if filepath.Ext(entry.Name()) != ".yaml" && filepath.Ext(entry.Name()) != ".yml" {
			continue
		}

		t.Run(entry.Name(), func(t *testing.T) {
			samplePath := filepath.Join(samplesDir, entry.Name())
			data, err := os.ReadFile(samplePath)
			if err != nil {
				t.Fatalf("Failed to read sample file: %v", err)
			}

			var sr SemanticRouter
			if err := yaml.Unmarshal(data, &sr); err != nil {
				t.Errorf("Failed to parse sample CR %s: %v", entry.Name(), err)
			}
		})
	}
}
