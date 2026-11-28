package controllers

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func TestKServePredictorServiceRequestsReferencingRouters(t *testing.T) {
	kserve := vllmv1alpha1.VLLMBackend{Type: "kserve", InferenceServiceName: "model"}
	makeRouter := func(namespace, name string, backends ...vllmv1alpha1.VLLMBackend) *vllmv1alpha1.SemanticRouter {
		router := &vllmv1alpha1.SemanticRouter{ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name}}
		for _, backend := range backends {
			router.Spec.VLLMEndpoints = append(router.Spec.VLLMEndpoints, vllmv1alpha1.VLLMEndpointSpec{Backend: backend})
		}
		return router
	}
	first := makeRouter("models", "first", kserve, kserve)
	second := makeRouter("models", "second", kserve)
	directService := vllmv1alpha1.VLLMBackend{Type: "service", Service: &vllmv1alpha1.ServiceBackend{Name: "model-predictor", Port: 80}}
	s := runtime.NewScheme()
	if err := vllmv1alpha1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	r := &SemanticRouterReconciler{Client: fake.NewClientBuilder().WithScheme(s).
		WithIndex(&vllmv1alpha1.SemanticRouter{}, kservePredictorServiceIndex, kservePredictorServiceNames).
		WithObjects(first, second,
			makeRouter("other", "cross-namespace", kserve),
			makeRouter("models", "another-inference-service", vllmv1alpha1.VLLMBackend{Type: "kserve", InferenceServiceName: "other"}),
			makeRouter("models", "explicit-service", directService),
		).Build()}

	service := &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: "model-predictor", Namespace: "models"}}
	requests := r.routersForKServePredictorService(t.Context(), service)
	if len(requests) != 2 {
		t.Fatalf("expected two referring SemanticRouters, got %v", requests)
	}
	matched := map[client.ObjectKey]bool{}
	for _, request := range requests {
		matched[request.NamespacedName] = true
	}
	if !matched[client.ObjectKeyFromObject(first)] || !matched[client.ObjectKeyFromObject(second)] {
		t.Fatalf("expected requests for first and second, got %v", requests)
	}
}
