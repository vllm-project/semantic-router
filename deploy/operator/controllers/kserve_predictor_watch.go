package controllers

import (
	"context"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

const (
	kservePredictorServiceIndex  = "spec.vllmEndpoints.kservePredictorService"
	kservePredictorServiceSuffix = "-predictor"
)

func kservePredictorServiceNames(obj client.Object) []string {
	router := obj.(*vllmv1alpha1.SemanticRouter)
	var names []string
	for _, endpoint := range router.Spec.VLLMEndpoints {
		if backend := endpoint.Backend; backend.Type == "kserve" && backend.InferenceServiceName != "" {
			names = append(names, backend.InferenceServiceName+kservePredictorServiceSuffix)
		}
	}
	return names
}

func (r *SemanticRouterReconciler) routersForKServePredictorService(ctx context.Context, service client.Object) []reconcile.Request {
	if !strings.HasSuffix(service.GetName(), kservePredictorServiceSuffix) {
		return nil
	}
	var routers vllmv1alpha1.SemanticRouterList
	if err := r.List(ctx, &routers, client.InNamespace(service.GetNamespace()), client.MatchingFields{
		kservePredictorServiceIndex: service.GetName(),
	}); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list SemanticRouters for KServe predictor Service", "service", client.ObjectKeyFromObject(service))
		return nil
	}
	requests := make([]reconcile.Request, 0, len(routers.Items))
	for i := range routers.Items {
		requests = append(requests, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(&routers.Items[i])})
	}
	return requests
}
