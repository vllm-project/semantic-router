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
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestValidateCreateRejectsMixedAuthoringSources(t *testing.T) {
	tests := []struct {
		name string
		sr   *SemanticRouter
	}{
		{
			name: "authoringConfig with vllmEndpoints",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: SemanticRouterSpec{
					AuthoringConfig: &apiextensionsv1.JSON{
						Raw: []byte(`{"version":"v0.1","providers":{"models":[{"name":"qwen3-4b","endpoints":[{"name":"primary","endpoint":"vllm-qwen.default.svc.cluster.local:8000"}]}],"default_model":"qwen3-4b"}}`),
					},
					VLLMEndpoints: []VLLMEndpointSpec{{
						Name:  "legacy-endpoint",
						Model: "qwen3-4b",
						Backend: VLLMBackend{
							Type:    "service",
							Service: &ServiceBackend{Name: "qwen-svc", Port: 8000},
						},
					}},
				},
			},
		},
		{
			name: "authoringConfig with overlapping config fields",
			sr: &SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: SemanticRouterSpec{
					AuthoringConfig: &apiextensionsv1.JSON{
						Raw: []byte(`{"version":"v0.1","providers":{"models":[{"name":"qwen3-4b","endpoints":[{"name":"primary","endpoint":"vllm-qwen.default.svc.cluster.local:8000"}]}],"default_model":"qwen3-4b"}}`),
					},
					Config: ConfigSpec{DefaultReasoningEffort: "medium"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.sr.ValidateCreate(); err == nil {
				t.Fatal("ValidateCreate() error = nil, want non-nil")
			}
		})
	}
}

func TestValidateUpdateAllowsAuthoringConfigOnly(t *testing.T) {
	old := &SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec:       SemanticRouterSpec{Replicas: func() *int32 { i := int32(1); return &i }()},
	}

	sr := &SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: SemanticRouterSpec{
			AuthoringConfig: &apiextensionsv1.JSON{
				Raw: []byte(`{"version":"v0.1","providers":{"models":[{"name":"qwen3-4b","endpoints":[{"name":"primary","endpoint":"vllm-qwen.default.svc.cluster.local:8000"}]}],"default_model":"qwen3-4b"}}`),
			},
		},
	}

	if _, err := sr.ValidateUpdate(old); err != nil {
		t.Fatalf("ValidateUpdate() error = %v, want nil", err)
	}
}
