package v1alpha1

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/defaulting"
	"sigs.k8s.io/yaml"
)

func TestGeneratedEmbeddingSoftMatchingDefaults(t *testing.T) {
	for _, path := range []string{
		"config/crd/bases/vllm.ai_semanticrouters.yaml",
		"bundle/manifests/vllm.ai_semanticrouters.yaml",
	} {
		t.Run(path, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("../..", path))
			if err != nil {
				t.Fatal(err)
			}
			var crd apiextensionsv1.CustomResourceDefinition
			if err := yaml.Unmarshal(data, &crd); err != nil {
				t.Fatal(err)
			}
			if len(crd.Spec.Versions) == 0 {
				t.Fatal("generated CRD has no versions")
			}
			for _, version := range crd.Spec.Versions {
				t.Run(version.Name, func(t *testing.T) {
					if version.Schema == nil || version.Schema.OpenAPIV3Schema == nil {
						t.Fatal("generated CRD has no structural schema")
					}
					var internal apiextensions.JSONSchemaProps
					if err := apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(
						version.Schema.OpenAPIV3Schema, &internal, nil,
					); err != nil {
						t.Fatal(err)
					}
					schema, err := structuralschema.NewStructural(&internal)
					if err != nil {
						t.Fatal(err)
					}
					for _, tc := range []struct {
						name     string
						provided bool
						value    bool
					}{
						{name: "omitted"},
						{name: "explicit_false", provided: true},
						{name: "explicit_true", provided: true, value: true},
					} {
						t.Run(tc.name, func(t *testing.T) {
							embedding := map[string]interface{}{}
							if tc.provided {
								embedding["enable_soft_matching"] = tc.value
							}
							object := map[string]interface{}{
								"spec": map[string]interface{}{
									"config": map[string]interface{}{
										"embedding_models": map[string]interface{}{
											"embedding_config": embedding,
										},
									},
								},
							}
							defaulting.Default(object, schema)
							if got := embedding["enable_soft_matching"]; got != tc.value {
								t.Fatalf("defaulted enable_soft_matching = %v, want %v", got, tc.value)
							}
							encoded, err := json.Marshal(embedding)
							if err != nil {
								t.Fatal(err)
							}
							var typed HNSWEmbeddingConfig
							if err := json.Unmarshal(encoded, &typed); err != nil {
								t.Fatal(err)
							}
							if typed.EnableSoftMatching != tc.value {
								t.Fatalf("typed enable_soft_matching = %v, want %v", typed.EnableSoftMatching, tc.value)
							}
						})
					}
				})
			}
		})
	}
}
