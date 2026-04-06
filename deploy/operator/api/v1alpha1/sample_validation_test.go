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
	projectRoot := filepath.Join("..", "..", "..")
	samplesDir := filepath.Join(projectRoot, "config", "samples")

	tests := []struct {
		name     string
		filename string
	}{
		{
			name:     "mmbert sample CR",
			filename: sampleMMBert,
		},
		{
			name:     "complexity routing sample CR",
			filename: sampleComplexity,
		},
		{
			name:     "simple sample CR",
			filename: sampleSimple,
		},
		{
			name:     "redis cache sample CR",
			filename: sampleRedisCache,
		},
		{
			name:     "milvus cache sample CR",
			filename: sampleMilvusCache,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			samplePath := filepath.Join(samplesDir, tt.filename)

			data, err := os.ReadFile(samplePath)
			if err != nil {
				t.Skipf("Sample file not found: %s (this is expected during development)", samplePath)
				return
			}

			var sr SemanticRouter
			if err := yaml.Unmarshal(data, &sr); err != nil {
				t.Errorf("Failed to unmarshal sample CR: %v", err)
				return
			}

			_, err = sr.ValidateCreate(context.Background(), &sr)
			if err != nil {
				t.Errorf("Sample CR validation failed: %v", err)
			}

			assertSampleSpecificContract(t, tt.filename, &sr)
		})
	}
}

func TestSampleCRsParseable(t *testing.T) {
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
