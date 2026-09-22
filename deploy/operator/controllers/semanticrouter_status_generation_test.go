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
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func TestStatusConditionsReportReconciledGeneration(t *testing.T) {
	tests := []struct {
		name      string
		missing   bool
		ready     int32
		phase     string
		condition string
	}{
		{name: "missing", missing: true, phase: "Pending", condition: typeAvailableSemanticRouter},
		{name: "pending", phase: "Pending", condition: typeAvailableSemanticRouter},
		{name: "progressing", ready: 1, phase: "Progressing", condition: typeProgressingSemanticRouter},
		{name: "running", ready: 2, phase: "Running", condition: typeAvailableSemanticRouter},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			if err := appsv1.AddToScheme(scheme); err != nil {
				t.Fatal(err)
			}
			if err := vllmv1alpha1.AddToScheme(scheme); err != nil {
				t.Fatal(err)
			}
			sr := &vllmv1alpha1.SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default", Generation: 42},
			}
			base := sr.DeepCopy()
			observed := make(chan vllmv1alpha1.SemanticRouterStatus, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				var response runtime.Object
				switch {
				case r.Method == http.MethodGet && r.URL.Path == "/apis/apps/v1/namespaces/default/deployments/router":
					if test.missing {
						w.WriteHeader(http.StatusNotFound)
						response = &metav1.Status{
							TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Status"},
							Status:   metav1.StatusFailure,
							Reason:   metav1.StatusReasonNotFound,
							Code:     http.StatusNotFound,
						}
					} else {
						response = &appsv1.Deployment{
							TypeMeta:   metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
							ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default"},
							Status:     appsv1.DeploymentStatus{Replicas: 2, ReadyReplicas: test.ready},
						}
					}
				case r.Method == http.MethodPatch && r.URL.Path == "/apis/vllm.ai/v1alpha1/namespaces/default/semanticrouters/router/status":
					var patch struct {
						Status vllmv1alpha1.SemanticRouterStatus `json:"status"`
					}
					if err := json.NewDecoder(r.Body).Decode(&patch); err != nil {
						http.Error(w, err.Error(), http.StatusBadRequest)
						return
					}
					observed <- patch.Status
					updated := base.DeepCopy()
					updated.APIVersion = vllmv1alpha1.GroupVersion.String()
					updated.Kind = "SemanticRouter"
					updated.Status = patch.Status
					response = updated
				default:
					http.Error(w, "unexpected API request", http.StatusNotFound)
					return
				}
				if err := json.NewEncoder(w).Encode(response); err != nil {
					t.Errorf("encode API fixture: %v", err)
				}
			}))
			t.Cleanup(server.Close)

			mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{appsv1.SchemeGroupVersion, vllmv1alpha1.GroupVersion})
			mapper.Add(appsv1.SchemeGroupVersion.WithKind("Deployment"), meta.RESTScopeNamespace)
			mapper.Add(vllmv1alpha1.GroupVersion.WithKind("SemanticRouter"), meta.RESTScopeNamespace)
			api, err := client.New(&rest.Config{Host: server.URL, Timeout: 5 * time.Second}, client.Options{Scheme: scheme, Mapper: mapper})
			if err != nil {
				t.Fatal(err)
			}
			reconciler := &SemanticRouterReconciler{Client: api, Scheme: scheme}
			if err := reconciler.updateStatus(t.Context(), sr, base); err != nil {
				t.Fatal(err)
			}

			select {
			case status := <-observed:
				condition := meta.FindStatusCondition(status.Conditions, test.condition)
				if condition == nil {
					t.Fatalf("status patch has no %s condition: %+v", test.condition, status)
				}
				t.Logf("phase=%s status_generation=%d condition=%s condition_generation=%d",
					status.Phase, status.ObservedGeneration, condition.Type, condition.ObservedGeneration)
				if status.Phase != test.phase || status.ObservedGeneration != sr.Generation {
					t.Errorf("status changed unexpectedly: %+v", status)
				}
				if condition.ObservedGeneration != sr.Generation {
					t.Errorf("condition generation = %d, want %d", condition.ObservedGeneration, sr.Generation)
				}
			default:
				t.Fatal("the actual Kubernetes client did not submit a status patch")
			}
		})
	}
}
