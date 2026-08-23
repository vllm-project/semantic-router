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
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/yaml"
)

func TestSamplesSelectBootstrapConfigMap(t *testing.T) {
	paths, err := filepath.Glob("../../config/samples/vllm.ai_v1alpha1_semanticrouter_*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) == 0 {
		t.Fatal("no SemanticRouter samples found")
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			documents := strings.Split(string(data), "\n---\n")
			if len(documents) != 2 {
				t.Fatalf("sample must contain one ConfigMap and one SemanticRouter, got %d documents", len(documents))
			}
			var configMap corev1.ConfigMap
			if err := yaml.Unmarshal([]byte(documents[0]), &configMap); err != nil {
				t.Fatal(err)
			}
			if configMap.Kind != "ConfigMap" || configMap.Immutable == nil || !*configMap.Immutable {
				t.Fatalf("first document must be an immutable ConfigMap: %#v", configMap.ObjectMeta)
			}
			var router SemanticRouter
			if err := yaml.Unmarshal([]byte(documents[1]), &router); err != nil {
				t.Fatal(err)
			}
			if router.Spec.Bootstrap.ConfigMapRef.Name == "" || router.Spec.Bootstrap.ConfigMapRef.Key == "" {
				t.Fatalf("sample must select spec.bootstrap.configMapRef: %#v", router.Spec.Bootstrap)
			}
			ref := router.Spec.Bootstrap.ConfigMapRef
			if ref.Name != configMap.Name {
				t.Fatalf("bootstrap reference %q does not select sample ConfigMap %q", ref.Name, configMap.Name)
			}
			if _, ok := configMap.Data[ref.Key]; !ok {
				t.Fatalf("bootstrap ConfigMap does not contain selected key %q", ref.Key)
			}
		})
	}
}
