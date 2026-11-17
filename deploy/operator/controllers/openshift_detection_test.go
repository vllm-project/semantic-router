package controllers

import (
	"context"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

// A failed route.openshift.io List proves nothing about the platform. It used
// to be cached as "this is OpenShift" for the process lifetime, which dropped
// the pod security context on vanilla clusters.
func TestTransientListErrorDoesNotPinPlatform(t *testing.T) {
	busy := true
	cl := fake.NewClientBuilder().WithInterceptorFuncs(interceptor.Funcs{
		List: func(_ context.Context, _ client.WithWatch, _ client.ObjectList, _ ...client.ListOption) error {
			if busy {
				return apierrors.NewTooManyRequests("busy", 1)
			}
			return &meta.NoResourceMatchError{PartialResource: schema.GroupVersionResource{
				Group: "route.openshift.io", Version: "v1", Resource: "routes",
			}}
		},
	}).Build()
	r := &SemanticRouterReconciler{Client: cl}

	if r.isRunningOnOpenShift(context.Background()) {
		t.Fatal("a transient API error must not be read as OpenShift")
	}
	if r.isOpenShift != nil {
		t.Fatal("a transient API error must not be cached as a verdict")
	}

	busy = false
	if r.isRunningOnOpenShift(context.Background()) {
		t.Fatal("a cluster without route.openshift.io is not OpenShift")
	}
	if r.isOpenShift == nil || *r.isOpenShift {
		t.Fatal("the NoMatch verdict should be cached as false")
	}
}

func TestNoMatchErrorIsCachedAsNotOpenShift(t *testing.T) {
	cl := fake.NewClientBuilder().WithInterceptorFuncs(interceptor.Funcs{
		List: func(_ context.Context, _ client.WithWatch, _ client.ObjectList, _ ...client.ListOption) error {
			return &meta.NoResourceMatchError{PartialResource: schema.GroupVersionResource{
				Group: "route.openshift.io", Version: "v1", Resource: "routes",
			}}
		},
	}).Build()
	r := &SemanticRouterReconciler{Client: cl}

	if r.isRunningOnOpenShift(context.Background()) {
		t.Fatal("no route API group means not OpenShift")
	}
	if r.isOpenShift == nil || *r.isOpenShift {
		t.Fatal("verdict should be cached as false")
	}
}
