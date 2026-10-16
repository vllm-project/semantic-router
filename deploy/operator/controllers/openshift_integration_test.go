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
	"testing"

	routev1 "github.com/openshift/api/route/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

// Disabling routes must delete the Route and stop reporting it in status;
// before the fix the status kept routesEnabled=true with the stale hostname.
func TestReconcileRouteClearsStatusWhenRoutesDisabled(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)
	_ = routev1.AddToScheme(s)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-router",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			OpenShift: &vllmv1alpha1.OpenShiftSpec{
				Routes: &vllmv1alpha1.RouteConfig{Enabled: false},
			},
		},
		Status: vllmv1alpha1.SemanticRouterStatus{
			OpenShiftFeatures: &vllmv1alpha1.OpenShiftFeaturesStatus{
				RoutesEnabled: true,
				RouteHostname: "sr.apps.example.com",
			},
		},
	}
	route := &routev1.Route{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sr.Name,
			Namespace: sr.Namespace,
		},
	}

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr, route).WithStatusSubresource(sr).Build()

	if err := reconcileRoute(context.Background(), cl, s, sr, true); err != nil {
		t.Fatalf("reconcileRoute() failed with routes disabled: %v", err)
	}

	if sr.Status.OpenShiftFeatures != nil {
		t.Errorf("OpenShiftFeatures: want cleared, got %+v", sr.Status.OpenShiftFeatures)
	}

	err := cl.Get(context.Background(), types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, &routev1.Route{})
	if !apierrors.IsNotFound(err) {
		t.Errorf("Route: want deleted, got err=%v", err)
	}
}

// The status must also clear when the feature is disabled on a cluster where
// no Route can exist, so a stale report cannot survive the spec change.
func TestReconcileRouteClearsStatusOnStandardKubernetesWhenDisabled(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-router",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			OpenShift: &vllmv1alpha1.OpenShiftSpec{
				Routes: &vllmv1alpha1.RouteConfig{Enabled: false},
			},
		},
		Status: vllmv1alpha1.SemanticRouterStatus{
			OpenShiftFeatures: &vllmv1alpha1.OpenShiftFeaturesStatus{
				RoutesEnabled: true,
				RouteHostname: "sr.apps.example.com",
			},
		},
	}

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr).WithStatusSubresource(sr).Build()

	if err := reconcileRoute(context.Background(), cl, s, sr, false); err != nil {
		t.Fatalf("reconcileRoute() failed on standard Kubernetes: %v", err)
	}

	if sr.Status.OpenShiftFeatures != nil {
		t.Errorf("OpenShiftFeatures: want cleared, got %+v", sr.Status.OpenShiftFeatures)
	}
}

func TestReconcileRouteSkipsCleanupOnStandardKubernetes(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-router",
			Namespace: "default",
		},
	}

	cl := fake.NewClientBuilder().WithScheme(s).Build()

	if err := reconcileRoute(context.Background(), cl, s, sr, false); err != nil {
		t.Fatalf("reconcileRoute() failed on standard Kubernetes without Route API: %v", err)
	}
}

func TestDeleteRouteIfExistsIgnoresUnavailableRouteAPI(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-router",
			Namespace: "default",
		},
	}

	cl := fake.NewClientBuilder().WithScheme(s).Build()

	if err := deleteRouteIfExists(context.Background(), cl, sr); err != nil {
		t.Fatalf("deleteRouteIfExists() should ignore unavailable Route API: %v", err)
	}
}
