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

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

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
