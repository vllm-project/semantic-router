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

func TestAuthoringSampleCRValidation(t *testing.T) {
	projectRoot := filepath.Join("..", "..")
	samplePath := filepath.Join(projectRoot, "config", "samples", "vllm.ai_v1alpha1_semanticrouter_authoring.yaml")

	data, err := os.ReadFile(samplePath)
	if err != nil {
		t.Fatalf("Failed to read authoring sample CR: %v", err)
	}

	var sr SemanticRouter
	if err := yaml.Unmarshal(data, &sr); err != nil {
		t.Fatalf("Failed to unmarshal authoring sample CR: %v", err)
	}

	if _, err := sr.ValidateCreate(); err != nil {
		t.Fatalf("Authoring sample validation failed: %v", err)
	}
	if sr.Spec.AuthoringConfig == nil {
		t.Fatal("authoring sample should set spec.authoringConfig")
	}
	if len(sr.Spec.VLLMEndpoints) != 0 {
		t.Fatal("authoring sample should not set spec.vllmEndpoints")
	}
	if sr.Spec.Config.API == nil {
		t.Fatal("authoring sample should keep non-overlapping operator config.api")
	}
}
