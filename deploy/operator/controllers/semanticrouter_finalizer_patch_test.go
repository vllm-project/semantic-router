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
	"encoding/json"
	"slices"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

// A full Update sends the spec as the Go types encode it, and the API server
// defaults whatever that encoding leaves out. A finalizer change must not be
// able to rewrite the spec, so it goes out as a merge patch of the metadata.
func TestFinalizerChangeSendsMetadataOnly(t *testing.T) {
	deleting := metav1.Now()
	tests := []struct {
		name           string
		meta           metav1.ObjectMeta
		wantDone       bool
		wantFinalizers []string
	}{
		{
			name:           "add",
			meta:           metav1.ObjectMeta{Name: "router", Namespace: "default"},
			wantFinalizers: []string{SemanticRouterFinalizer},
		},
		{
			name: "remove",
			meta: metav1.ObjectMeta{
				Name: "router", Namespace: "default",
				DeletionTimestamp: &deleting,
				Finalizers:        []string{SemanticRouterFinalizer},
			},
			wantDone: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			if err := vllmv1alpha1.AddToScheme(scheme); err != nil {
				t.Fatal(err)
			}
			stored := &vllmv1alpha1.SemanticRouter{
				ObjectMeta: test.meta,
				Spec: vllmv1alpha1.SemanticRouterSpec{Config: vllmv1alpha1.ConfigSpec{
					ResponseCache: &vllmv1alpha1.SemanticCacheConfig{Enabled: ptr.To(false)},
				}},
			}

			type sentPatch struct {
				kind types.PatchType
				body []byte
			}
			var patches []sentPatch
			api := fake.NewClientBuilder().WithScheme(scheme).WithObjects(stored).WithInterceptorFuncs(interceptor.Funcs{
				Update: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.UpdateOption) error {
					t.Errorf("finalizer change sent a full Update of %T", obj)
					return c.Update(ctx, obj, opts...)
				},
				Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
					body, err := patch.Data(obj)
					if err != nil {
						return err
					}
					patches = append(patches, sentPatch{kind: patch.Type(), body: body})
					return c.Patch(ctx, obj, patch, opts...)
				},
			}).Build()

			current := &vllmv1alpha1.SemanticRouter{}
			if err := api.Get(t.Context(), client.ObjectKeyFromObject(stored), current); err != nil {
				t.Fatal(err)
			}
			reconciler := &SemanticRouterReconciler{Client: api, Scheme: scheme}
			done, err := reconciler.handleFinalizerFlow(t.Context(), current)
			if err != nil || done != test.wantDone {
				t.Fatalf("handleFinalizerFlow: done=%t error=%v, want done=%t", done, err, test.wantDone)
			}

			if len(patches) != 1 {
				t.Fatalf("finalizer change sent %d patches, want 1", len(patches))
			}
			if patches[0].kind != types.MergePatchType {
				t.Errorf("patch type = %s, want %s", patches[0].kind, types.MergePatchType)
			}
			var body map[string]json.RawMessage
			if err := json.Unmarshal(patches[0].body, &body); err != nil {
				t.Fatalf("decode patch %s: %v", patches[0].body, err)
			}
			if len(body) != 1 || body["metadata"] == nil {
				t.Fatalf("patch carries more than metadata: %s", patches[0].body)
			}
			var metadata struct {
				Finalizers      []string `json:"finalizers"`
				ResourceVersion string   `json:"resourceVersion"`
			}
			if err := json.Unmarshal(body["metadata"], &metadata); err != nil {
				t.Fatalf("decode patch metadata %s: %v", body["metadata"], err)
			}
			if !slices.Equal(metadata.Finalizers, test.wantFinalizers) {
				t.Errorf("patched finalizers = %v, want %v", metadata.Finalizers, test.wantFinalizers)
			}
			if metadata.ResourceVersion == "" {
				t.Errorf("patch has no resourceVersion, so a concurrent finalizer change is overwritten: %s", patches[0].body)
			}
		})
	}
}
