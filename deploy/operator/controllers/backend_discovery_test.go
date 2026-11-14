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
	"context"
	"errors"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDiscoverVLLMBackendsAggregatesFailures(t *testing.T) {
	s := backendDiscoveryTestScheme(t)
	r := &SemanticRouterReconciler{Client: fake.NewClientBuilder().WithScheme(s).Build()}

	models, err := discoverVLLMBackends(context.Background(), r.Client, []vllmv1alpha1.VLLMEndpointSpec{
		{
			Name:  "missing-kserve",
			Model: "llama-3",
			Backend: vllmv1alpha1.VLLMBackend{
				Type:                 "kserve",
				InferenceServiceName: "missing-isvc",
			},
		},
		{
			Name:  "missing-llamastack",
			Model: "llama-3",
			Backend: vllmv1alpha1.VLLMBackend{
				Type:            "llamastack",
				DiscoveryLabels: map[string]string{"app": "llama-stack"},
			},
		},
	}, "default")
	if models != nil {
		t.Fatalf("failed discovery returned models: %#v", models)
	}

	var discoveryErr *backendDiscoveryError
	if !errors.As(err, &discoveryErr) {
		t.Fatalf("discoverVLLMBackends() error = %v, want backendDiscoveryError", err)
	}
	if len(discoveryErr.Failures) != 2 {
		t.Fatalf("discovery failures = %#v, want two failures", discoveryErr.Failures)
	}
	for _, name := range []string{"missing-kserve", "missing-llamastack"} {
		if !strings.Contains(err.Error(), name) {
			t.Fatalf("discovery error %q does not name %q", err, name)
		}
	}
}

func TestDiscoverVLLMBackendsRejectsPartialFailure(t *testing.T) {
	s := backendDiscoveryTestScheme(t)
	r := &SemanticRouterReconciler{Client: fake.NewClientBuilder().WithScheme(s).Build()}

	models, err := discoverVLLMBackends(context.Background(), r.Client, []vllmv1alpha1.VLLMEndpointSpec{
		{
			Name:  "available-service",
			Model: "available-model",
			Backend: vllmv1alpha1.VLLMBackend{
				Type: "service",
				Service: &vllmv1alpha1.ServiceBackend{
					Name: "model-service",
					Port: 8000,
				},
			},
		},
		{
			Name:  "missing-kserve",
			Model: "missing-model",
			Backend: vllmv1alpha1.VLLMBackend{
				Type:                 "kserve",
				InferenceServiceName: "missing-isvc",
			},
		},
	}, "default")
	if models != nil {
		t.Fatalf("partial failure returned models: %#v", models)
	}
	var discoveryErr *backendDiscoveryError
	if !errors.As(err, &discoveryErr) {
		t.Fatalf("discoverVLLMBackends() error = %v, want backendDiscoveryError", err)
	}
}

func TestReconcileConfigMapPreservesLastGoodConfigOnDiscoveryFailure(t *testing.T) {
	s := backendDiscoveryTestScheme(t)
	sr := backendDiscoveryTestRouter()
	previous := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "router-config", Namespace: "default"},
		Data: map[string]string{
			"config.yaml":   "previous-config",
			"tools_db.json": "[]",
		},
	}
	r := &SemanticRouterReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).WithObjects(sr, previous).Build(),
		Scheme: s,
	}

	if err := r.reconcileConfigMap(context.Background(), sr); err == nil {
		t.Fatal("reconcileConfigMap() succeeded despite a missing discovered backend")
	}

	got := &corev1.ConfigMap{}
	if err := r.Get(context.Background(), client.ObjectKeyFromObject(previous), got); err != nil {
		t.Fatal(err)
	}
	if got.Data["config.yaml"] != "previous-config" || got.Data["tools_db.json"] != "[]" {
		t.Fatalf("previous ConfigMap was changed: %#v", got.Data)
	}
}

func TestReconcileConfigMapRecoversWhenKServeAppears(t *testing.T) {
	s := backendDiscoveryTestScheme(t)
	sr := backendDiscoveryTestRouter()
	r := &SemanticRouterReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).WithObjects(sr).Build(),
		Scheme: s,
	}
	ctx := context.Background()

	if err := r.reconcileConfigMap(ctx, sr); err == nil {
		t.Fatal("initial reconcileConfigMap() succeeded without the KServe resource")
	}

	inferenceService := &unstructured.Unstructured{}
	inferenceService.SetGroupVersionKind(schema.GroupVersionKind{
		Group: "serving.kserve.io", Version: "v1beta1", Kind: "InferenceService",
	})
	inferenceService.SetName("missing-isvc")
	inferenceService.SetNamespace("default")
	if err := r.Create(ctx, inferenceService); err != nil {
		t.Fatal(err)
	}

	if err := r.reconcileConfigMap(ctx, sr); err != nil {
		t.Fatalf("reconcileConfigMap() after dependency recovery failed: %v", err)
	}
	cm := &corev1.ConfigMap{}
	if err := r.Get(ctx, client.ObjectKey{Name: "router-config", Namespace: "default"}, cm); err != nil {
		t.Fatal(err)
	}
	parsed, err := routerconfig.ParseYAMLBytes([]byte(cm.Data["config.yaml"]))
	if err != nil {
		t.Fatalf("recovered config is not valid Router config: %v", err)
	}
	if parsed.DefaultModel != "llama-3" || len(parsed.GetEndpointsForModel("llama-3")) != 1 {
		t.Fatalf("recovered config default=%q endpoints=%#v", parsed.DefaultModel, parsed.GetEndpointsForModel("llama-3"))
	}
}

func TestBackendDiscoveryStatusIsReportedAndCleared(t *testing.T) {
	s := backendDiscoveryTestScheme(t)
	sr := backendDiscoveryTestRouter()
	r := &SemanticRouterReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).WithObjects(sr).WithStatusSubresource(sr).Build(),
		Scheme: s,
	}
	discoveryErr := &backendDiscoveryError{Failures: []backendDiscoveryFailure{{
		EndpointName: "missing-kserve",
		Dependency:   "InferenceService default/missing-isvc",
		Err:          errors.New("not found"),
	}}}

	if err := r.updateBackendDiscoveryStatus(context.Background(), sr, discoveryErr); err != nil {
		t.Fatal(err)
	}
	got := &vllmv1alpha1.SemanticRouter{}
	if err := r.Get(context.Background(), client.ObjectKeyFromObject(sr), got); err != nil {
		t.Fatal(err)
	}
	condition := meta.FindStatusCondition(got.Status.Conditions, typeDegradedSemanticRouter)
	if condition == nil || condition.Status != metav1.ConditionTrue || condition.Reason != "BackendDiscoveryFailed" ||
		!strings.Contains(condition.Message, "InferenceService default/missing-isvc") || got.Status.Phase != "Degraded" {
		t.Fatalf("unexpected degraded status: %+v", got.Status)
	}

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: sr.Name, Namespace: sr.Namespace},
		Status:     appsv1.DeploymentStatus{Replicas: 1, ReadyReplicas: 1},
	}
	if err := r.Create(context.Background(), deployment); err != nil {
		t.Fatal(err)
	}
	base := got.DeepCopy()
	if err := r.updateStatus(context.Background(), got, base); err != nil {
		t.Fatal(err)
	}
	if err := r.Get(context.Background(), client.ObjectKeyFromObject(sr), got); err != nil {
		t.Fatal(err)
	}
	if meta.FindStatusCondition(got.Status.Conditions, typeDegradedSemanticRouter) != nil {
		t.Fatalf("degraded condition survived successful status update: %+v", got.Status)
	}
}

func backendDiscoveryTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	if err := scheme.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	if err := vllmv1alpha1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	return s
}

func backendDiscoveryTestRouter() *vllmv1alpha1.SemanticRouter {
	return &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default"},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			VLLMEndpoints: []vllmv1alpha1.VLLMEndpointSpec{{
				Name:  "missing-kserve",
				Model: "llama-3",
				Backend: vllmv1alpha1.VLLMBackend{
					Type:                 "kserve",
					InferenceServiceName: "missing-isvc",
				},
			}},
		},
	}
}
