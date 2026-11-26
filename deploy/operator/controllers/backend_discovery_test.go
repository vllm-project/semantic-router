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

package controllers

import (
	"testing"

	"gopkg.in/yaml.v3"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestGenerateConfigYAMLUsesKServePredictorService(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		inferenceServiceName string
		ports                []corev1.ServicePort
		wantEndpoint         string
		wantProtocol         string
	}{
		{
			name: "standard HTTP Service",
			ports: []corev1.ServicePort{{
				Name: "inference-model-predictor", Protocol: corev1.ProtocolTCP, Port: 80,
				TargetPort: intstr.FromInt32(8080),
			}},
			wantEndpoint: "inference-model-predictor.test.svc.cluster.local:80",
			wantProtocol: "http",
		},
		{
			name:                 "standard HTTP Service for an InferenceService named https-model",
			inferenceServiceName: "https-model",
			ports: []corev1.ServicePort{{
				Name: "https-model-predictor", Protocol: corev1.ProtocolTCP, Port: 80,
				TargetPort: intstr.FromInt32(8080),
			}},
			wantEndpoint: "https-model-predictor.test.svc.cluster.local:80",
			wantProtocol: "http",
		},
		{
			name:                 "standard HTTP Service for an InferenceService named grpc-model",
			inferenceServiceName: "grpc-model",
			ports: []corev1.ServicePort{{
				Name: "grpc-model-predictor", Protocol: corev1.ProtocolTCP, Port: 80,
				TargetPort: intstr.FromInt32(8080),
			}},
			wantEndpoint: "grpc-model-predictor.test.svc.cluster.local:80",
			wantProtocol: "http",
		},
		{
			name: "legacy TLS port alongside HTTP",
			ports: []corev1.ServicePort{
				{Name: "http", Port: 80},
				{Name: "tls", Port: 8443},
			},
			wantEndpoint: "inference-model-predictor.test.svc.cluster.local:8443",
			wantProtocol: "https",
		},
		{
			name:         "TLS on a custom named port",
			ports:        []corev1.ServicePort{{Name: "https-inference", Port: 9443}},
			wantEndpoint: "inference-model-predictor.test.svc.cluster.local:9443",
			wantProtocol: "https",
		},
		{
			name:         "declared HTTPS on a custom Service port",
			ports:        []corev1.ServicePort{{Name: "predictor", AppProtocol: ptr.To("https"), Port: 4443}},
			wantEndpoint: "inference-model-predictor.test.svc.cluster.local:4443",
			wantProtocol: "https",
		},
		{
			name:         "declared HTTP on a conventional TLS port",
			ports:        []corev1.ServicePort{{Name: "https", AppProtocol: ptr.To("http"), Port: 8443}},
			wantEndpoint: "inference-model-predictor.test.svc.cluster.local:8443",
			wantProtocol: "http",
		},
		{
			name: "HTTP after other protocols",
			ports: []corev1.ServicePort{
				{Name: "dns", Protocol: corev1.ProtocolUDP, Port: 443},
				{Name: "grpc", AppProtocol: ptr.To("kubernetes.io/h2c"), Port: 8443},
				{Name: "predictor", Port: 8181},
			},
			wantEndpoint: "inference-model-predictor.test.svc.cluster.local:8181",
			wantProtocol: "http",
		},
		{
			name:  "no HTTP port",
			ports: []corev1.ServicePort{{Name: "grpc", AppProtocol: ptr.To("kubernetes.io/h2c"), Port: 80}},
		},
		{name: "predictor Service only in another namespace"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			name := tc.inferenceServiceName
			if name == "" {
				name = "inference-model"
			}
			testKServePredictorConfig(t, name, tc.ports, tc.wantEndpoint, tc.wantProtocol)
		})
	}
}

func testKServePredictorConfig(t *testing.T, inferenceServiceName string, ports []corev1.ServicePort, wantEndpoint, wantProtocol string) {
	t.Helper()

	isvc := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "serving.kserve.io/v1beta1",
		"kind":       "InferenceService",
		"metadata":   map[string]interface{}{"name": inferenceServiceName, "namespace": "test"},
		"status":     map[string]interface{}{"url": "https://inference-model.example.test"},
	}}
	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "test-router", Namespace: "test"},
		Spec: vllmv1alpha1.SemanticRouterSpec{VLLMEndpoints: []vllmv1alpha1.VLLMEndpointSpec{{
			Name: "predictor", Model: "inference-model",
			Backend: vllmv1alpha1.VLLMBackend{Type: "kserve", InferenceServiceName: isvc.GetName()},
		}}},
	}
	predictorName := inferenceServiceName + "-predictor"
	otherNamespacePredictor := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: predictorName, Namespace: "other"},
		Spec:       corev1.ServiceSpec{Ports: []corev1.ServicePort{{Name: "https", Port: 443}}},
	}
	clientBuilder := fake.NewClientBuilder().WithObjects(isvc, otherNamespacePredictor)
	if ports != nil {
		clientBuilder.WithObjects(&corev1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: predictorName, Namespace: "test"},
			Spec:       corev1.ServiceSpec{Ports: ports},
		})
	}
	r := &SemanticRouterReconciler{Client: clientBuilder.Build()}
	configYAML, err := r.generateConfigYAML(t.Context(), sr)
	if err != nil {
		t.Fatal(err)
	}
	var config routerconfig.CanonicalConfig
	if err := yaml.Unmarshal([]byte(configYAML), &config); err != nil {
		t.Fatal(err)
	}
	models := config.Providers.Models
	if wantEndpoint == "" {
		if len(models) != 0 {
			t.Fatalf("expected discovery to omit the model, got %+v", models)
		}
		return
	}
	if len(models) != 1 || len(models[0].BackendRefs) != 1 {
		t.Fatalf("expected one discovered backend, got %+v", models)
	}
	ref := models[0].BackendRefs[0]
	if ref.Endpoint != wantEndpoint || ref.Protocol != wantProtocol {
		t.Errorf("expected %s://%s, got %s://%s", wantProtocol, wantEndpoint, ref.Protocol, ref.Endpoint)
	}
}
