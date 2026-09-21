package k8s

import (
	"context"
	"strings"
	"testing"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestReconcileConversionFailureReportsCurrentGenerationAndRecovers(t *testing.T) {
	for _, tc := range []struct {
		name, plugin, configuration, message string
	}{
		{"system prompt mode typo", "system_prompt", `{"system_prompt":"Say hello.","mode":"append"}`, "system_prompt mode must be"},
		{"response cache value type", "response_cache", `{"enabled":"yes"}`, "cannot unmarshal string"},
		{"empty header configuration", "header_mutation", `{}`, "must specify at least one"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			pool := buildConflictPool("default", "pool")
			pool.Generation = 1
			route := buildConflictRoute("default", "route")
			route.Generation = 1
			r := buildConflictReconciler(t, "default", pool, route)
			activations := 0
			var serving *config.RouterConfig
			r.onConfigUpdate = func(_ context.Context, candidate *config.RouterConfig) error {
				activations++
				serving = candidate
				return nil
			}
			if err := r.reconcile(ctx); err != nil {
				t.Fatal(err)
			}
			previous := serving
			key := client.ObjectKeyFromObject(route)
			if err := r.client.Get(ctx, key, route); err != nil {
				t.Fatal(err)
			}
			// Plugin configuration is arbitrary JSON in the CRD schema. These
			// ordinary authoring mistakes reach the real converter after reference
			// validation, rather than being rejected by API schema validation.
			route.Generation = 2
			route.Spec.Decisions[0].Plugins = []v1alpha1.DecisionPlugin{{
				Type: tc.plugin, Configuration: &runtime.RawExtension{Raw: []byte(tc.configuration)},
			}}
			if err := r.client.Update(ctx, route); err != nil {
				t.Fatal(err)
			}
			reconcileErr := r.reconcile(ctx)
			if reconcileErr == nil || !strings.Contains(reconcileErr.Error(), tc.message) {
				t.Fatalf("conversion rejection = %v, want %q", reconcileErr, tc.message)
			}
			if activations != 1 || serving != previous || r.lastRoute.Generation != 1 {
				t.Fatal("rejected conversion changed the serving generation or was acknowledged")
			}
			if err := r.client.Get(ctx, key, route); err != nil {
				t.Fatal(err)
			}
			if err := r.client.Get(ctx, client.ObjectKeyFromObject(pool), pool); err != nil {
				t.Fatal(err)
			}
			for _, status := range []struct {
				name       string
				generation int64
				observed   int64
				conditions []metav1.Condition
			}{
				{"pool", pool.Generation, pool.Status.ObservedGeneration, pool.Status.Conditions},
				{"route", route.Generation, route.Status.ObservedGeneration, route.Status.Conditions},
			} {
				ready := apimeta.FindStatusCondition(status.conditions, "Ready")
				if ready == nil || ready.Status != metav1.ConditionFalse || ready.Reason != "ValidationFailed" ||
					ready.ObservedGeneration != status.generation || status.observed != status.generation ||
					!strings.Contains(ready.Message, tc.message) {
					t.Errorf("%s must report current-generation conversion failure: observed=%d Ready=%+v", status.name, status.observed, ready)
				}
			}
			route.Generation = 3
			route.Spec.Decisions[0].Plugins = nil
			if err := r.client.Update(ctx, route); err != nil {
				t.Fatal(err)
			}
			if err := r.reconcile(ctx); err != nil {
				t.Fatal(err)
			}
			if activations != 2 || serving == previous || r.lastRoute.Generation != 3 {
				t.Fatal("corrected candidate was not activated")
			}
			if err := r.client.Get(ctx, key, route); err != nil {
				t.Fatal(err)
			}
			ready := apimeta.FindStatusCondition(route.Status.Conditions, "Ready")
			if ready == nil || ready.Status != metav1.ConditionTrue || ready.ObservedGeneration != 3 || route.Status.ObservedGeneration != 3 {
				t.Fatalf("recovered generation not ready: %+v", route.Status)
			}
		})
	}
}
