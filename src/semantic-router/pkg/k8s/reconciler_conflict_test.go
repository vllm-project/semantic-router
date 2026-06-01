package k8s

import (
	"context"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestReconcileMultipleCRsReportsConflict is the regression artifact for
// #1908: before the fix, a second IntelligentRoute (or IntelligentPool) made
// reconcile return early without ever writing status, so every CR sat with an
// empty status indefinitely. The reconcile loop must instead mark each
// conflicting CR Ready=False/Reason=Conflict.
func TestReconcileMultipleCRsReportsConflict(t *testing.T) {
	const namespace = "default"

	t.Run("MultipleRoutes", func(t *testing.T) {
		pool := buildConflictPool(namespace, "pool-a")
		routeA := buildConflictRoute(namespace, "route-a")
		routeB := buildConflictRoute(namespace, "route-b")

		reconciler := buildConflictReconciler(t, namespace, pool, routeA, routeB)

		err := reconciler.reconcile(context.Background())
		if err == nil {
			t.Fatal("reconcile: expected a conflict error with multiple routes, got nil")
		}
		if !strings.Contains(err.Error(), "IntelligentRoutes") {
			t.Errorf("conflict error should name the conflicting kind, got: %v", err)
		}

		assertConflictRouteStatus(t, reconciler, namespace, "route-a")
		assertConflictRouteStatus(t, reconciler, namespace, "route-b")
	})

	t.Run("MultiplePools", func(t *testing.T) {
		poolA := buildConflictPool(namespace, "pool-a")
		poolB := buildConflictPool(namespace, "pool-b")
		route := buildConflictRoute(namespace, "route-a")

		reconciler := buildConflictReconciler(t, namespace, poolA, poolB, route)

		err := reconciler.reconcile(context.Background())
		if err == nil {
			t.Fatal("reconcile: expected a conflict error with multiple pools, got nil")
		}
		if !strings.Contains(err.Error(), "IntelligentPools") {
			t.Errorf("conflict error should name the conflicting kind, got: %v", err)
		}

		assertConflictPoolStatus(t, reconciler, namespace, "pool-a")
		assertConflictPoolStatus(t, reconciler, namespace, "pool-b")
	})
}

// TestReconcileSingleCRUnchanged guards the happy path: exactly one pool and
// one route must still reconcile to Ready=True. This is the regression canary
// ensuring the #1908 fix did not alter single-CR behavior.
func TestReconcileSingleCRUnchanged(t *testing.T) {
	const namespace = "default"

	pool := buildConflictPool(namespace, "pool-a")
	route := buildConflictRoute(namespace, "route-a")

	reconciler := buildConflictReconciler(t, namespace, pool, route)

	if err := reconciler.reconcile(context.Background()); err != nil {
		t.Fatalf("reconcile: expected success with a single pool and route, got: %v", err)
	}

	assertReadyConditionByName(t, reconciler, namespace, "pool", "pool-a", metav1.ConditionTrue, "Ready")
	assertReadyConditionByName(t, reconciler, namespace, "route", "route-a", metav1.ConditionTrue, "Ready")
}

func buildConflictPool(namespace, name string) *v1alpha1.IntelligentPool {
	return &v1alpha1.IntelligentPool{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1alpha1.IntelligentPoolSpec{
			DefaultModel: "deepseek-v3",
			Models: []v1alpha1.ModelConfig{
				{Name: "deepseek-v3"},
			},
		},
	}
}

func buildConflictRoute(namespace, name string) *v1alpha1.IntelligentRoute {
	return &v1alpha1.IntelligentRoute{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1alpha1.IntelligentRouteSpec{
			Signals: v1alpha1.Signals{
				Embeddings: []v1alpha1.EmbeddingSignal{
					{
						Name:              "rule_" + name,
						Threshold:         0.7,
						Candidates:        []string{"anchor phrase"},
						AggregationMethod: "max",
						QueryModality:     "text",
					},
				},
			},
			Decisions: []v1alpha1.Decision{
				{
					Name:     "decision_" + name,
					Priority: 100,
					Signals: v1alpha1.SignalCombination{
						Operator: "AND",
						Conditions: []v1alpha1.SignalCondition{
							{Type: "embedding", Name: "rule_" + name},
						},
					},
					ModelRefs: []v1alpha1.ModelRef{
						{Model: "deepseek-v3"},
					},
				},
			},
		},
	}
}

func buildConflictReconciler(t *testing.T, namespace string, objs ...client.Object) *Reconciler {
	t.Helper()

	scheme := runtime.NewScheme()
	if err := v1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme: %v", err)
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&v1alpha1.IntelligentPool{}, &v1alpha1.IntelligentRoute{}).
		WithObjects(objs...).
		Build()

	return &Reconciler{
		client:    fakeClient,
		scheme:    scheme,
		namespace: namespace,
		converter: NewCRDConverter(),
		staticConfig: &config.RouterConfig{
			ConfigSource: config.ConfigSourceKubernetes,
			InlineModels: config.InlineModels{
				EmbeddingModels: config.EmbeddingModels{
					EmbeddingConfig: config.HNSWConfig{
						ModelType: "qwen3",
					},
				},
			},
		},
		onConfigUpdate: func(*config.RouterConfig) error { return nil },
	}
}

func assertConflictRouteStatus(t *testing.T, reconciler *Reconciler, namespace, name string) {
	t.Helper()
	assertReadyConditionByName(t, reconciler, namespace, "route", name, metav1.ConditionFalse, "Conflict")
}

func assertConflictPoolStatus(t *testing.T, reconciler *Reconciler, namespace, name string) {
	t.Helper()
	assertReadyConditionByName(t, reconciler, namespace, "pool", name, metav1.ConditionFalse, "Conflict")
}

func assertReadyConditionByName(
	t *testing.T,
	reconciler *Reconciler,
	namespace, kind, name string,
	wantStatus metav1.ConditionStatus,
	wantReason string,
) {
	t.Helper()

	var conditions []metav1.Condition
	ctx := context.Background()
	key := client.ObjectKey{Namespace: namespace, Name: name}

	switch kind {
	case "pool":
		var pool v1alpha1.IntelligentPool
		if err := reconciler.client.Get(ctx, key, &pool); err != nil {
			t.Fatalf("Get pool %s: %v", name, err)
		}
		conditions = pool.Status.Conditions
	case "route":
		var route v1alpha1.IntelligentRoute
		if err := reconciler.client.Get(ctx, key, &route); err != nil {
			t.Fatalf("Get route %s: %v", name, err)
		}
		conditions = route.Status.Conditions
	default:
		t.Fatalf("unknown kind %q", kind)
	}

	assertReadyCondition(t, kind+"/"+name, conditions, wantStatus, wantReason)
}
