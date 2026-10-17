package k8s

import (
	"context"
	"strings"
	"testing"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestReconcileDecisionRuleLimitRetainsActiveGeneration(t *testing.T) {
	ctx := context.Background()
	pool, route := buildConflictPool("default", "pool"), buildConflictRoute("default", "route")
	pool.Generation, route.Generation = 1, 1
	r := buildConflictReconciler(t, "default", pool, route)
	maxNodes := 2
	r.staticConfig.DecisionRuleLimits.MaxNodes = &maxNodes
	activations := 0
	var active *config.RouterConfig
	r.onConfigUpdate = func(_ context.Context, candidate *config.RouterConfig) error {
		activations++
		active = candidate
		return nil
	}
	if err := r.reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	previous := active
	key := client.ObjectKeyFromObject(route)
	if err := r.client.Get(ctx, key, route); err != nil {
		t.Fatal(err)
	}
	route.Generation = 2
	rules := &route.Spec.Decisions[0].Signals
	rules.Conditions = append(rules.Conditions, rules.Conditions[0])
	if err := r.client.Update(ctx, route); err != nil {
		t.Fatal(err)
	}
	err := r.reconcile(ctx)
	if err == nil || !strings.Contains(err.Error(), "node count 3 exceeds max_nodes=2") {
		t.Fatalf("expected rule budget rejection, got %v", err)
	}
	if activations != 1 || active != previous || r.lastRoute.Generation != 1 {
		t.Fatal("oversized candidate changed the active generation")
	}
	if err := r.client.Get(ctx, key, route); err != nil {
		t.Fatal(err)
	}
	ready := apimeta.FindStatusCondition(route.Status.Conditions, "Ready")
	if ready == nil || ready.Status != metav1.ConditionFalse || ready.Reason != "ValidationFailed" || ready.ObservedGeneration != 2 || !strings.Contains(ready.Message, "max_nodes=2") {
		t.Fatalf("expected current-generation observable rejection, got %+v", ready)
	}
	route.Generation = 3
	route.Spec.Decisions[0].Signals.Conditions = route.Spec.Decisions[0].Signals.Conditions[:1]
	if err := r.client.Update(ctx, route); err != nil {
		t.Fatal(err)
	}
	if err := r.reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	if activations != 2 || active == previous || r.lastRoute.Generation != 3 {
		t.Fatal("corrected candidate was not activated")
	}
}
