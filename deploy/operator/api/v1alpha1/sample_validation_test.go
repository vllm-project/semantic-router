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
	"context"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/apimachinery/pkg/util/yaml"
)

func TestSampleCRValidation(t *testing.T) {
	// Get the project root directory
	projectRoot := filepath.Join("..", "..")
	samplesDir := filepath.Join(projectRoot, "config", "samples")

	tests := []struct {
		name     string
		filename string
	}{
		{
			name:     "mmbert sample CR",
			filename: "vllm.ai_v1alpha1_semanticrouter_embedding.yaml",
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
				t.Fatalf("Required sample file not readable: %s: %v", samplePath, err)
				return
			}

			// Parse the YAML into a SemanticRouter object
			var sr SemanticRouter
			if err = yaml.Unmarshal(data, &sr); err != nil {
				t.Errorf("Failed to unmarshal sample CR: %v", err)
				return
			}

			// Validate the CR using webhook validation
			_, err = sr.ValidateCreate(context.Background(), &sr)
			if err != nil {
				t.Errorf("Sample CR validation failed: %v", err)
			}

			// Additional checks for specific samples
			if tt.filename == "vllm.ai_v1alpha1_semanticrouter_embedding.yaml" {
				if cache := sr.Spec.Config.ResponseCache; cache == nil || cache.Embedding == nil || cache.Embedding.Dimension != 256 {
					t.Error("embedding sample should set the response cache's embedding dimension")
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
					if rule.Hard != nil || rule.Easy != nil || rule.Composer != nil {
						t.Errorf("complexity rule %s writes example banks, which the decision model does not read", rule.Name)
					}
				}
			}

			if tt.filename == "vllm.ai_v1alpha1_semanticrouter_redis_cache.yaml" {
				if sr.Spec.Config.ResponseCache != nil {
					if sr.Spec.Config.ResponseCache.BackendType != "redis" {
						t.Errorf("redis cache sample backend_type = %v, want redis", sr.Spec.Config.ResponseCache.BackendType)
					}
				}
			}

			if tt.filename == "vllm.ai_v1alpha1_semanticrouter_milvus_cache.yaml" {
				if sr.Spec.Config.ResponseCache != nil {
					if sr.Spec.Config.ResponseCache.BackendType != "milvus" {
						t.Errorf("milvus cache sample backend_type = %v, want milvus", sr.Spec.Config.ResponseCache.BackendType)
					}
				}
			}
		})
	}
}

func TestSampleCRsParseable(t *testing.T) {
	// Test that all sample CRs can be parsed without errors
	projectRoot := filepath.Join("..", "..")
	samplesDir := filepath.Join(projectRoot, "config", "samples")

	entries, err := os.ReadDir(samplesDir)
	if err != nil {
		t.Fatalf("Required samples directory not readable: %s: %v", samplesDir, err)
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
