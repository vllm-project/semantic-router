package k8s

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestReconcileAcknowledgesActivationAndRetriesFailure(t *testing.T) {
	r := buildConflictReconciler(t, "default", buildConflictPool("default", "pool"), buildConflictRoute("default", "route"))
	calls := 0
	r.onConfigUpdate = func(context.Context, *config.RouterConfig) error {
		calls++
		assertReadyConditionByName(t, r, "default", "pool", "pool", metav1.ConditionFalse, "Activating")
		if calls == 1 {
			return errors.New("candidate warmup failed")
		}
		return nil
	}
	if err := r.reconcile(context.Background()); err == nil {
		t.Fatal("failed runtime was acknowledged")
	}
	assertReadyConditionByName(t, r, "default", "route", "route", metav1.ConditionFalse, "ActivationFailed")
	if err := r.reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	assertReadyConditionByName(t, r, "default", "route", "route", metav1.ConditionTrue, "Ready")
	if calls != 2 {
		t.Fatalf("calls = %d", calls)
	}
}

func TestReconcileRepairsStatusWithoutReactivation(t *testing.T) {
	r := buildConflictReconciler(t, "default", buildConflictPool("default", "pool"), buildConflictRoute("default", "route"))
	calls := 0
	r.onConfigUpdate = func(context.Context, *config.RouterConfig) error { calls++; return nil }
	underlying := r.client
	failStatus := true
	r.client = interceptor.NewClient(underlying.(client.WithWatch), interceptor.Funcs{
		SubResourceUpdate: func(ctx context.Context, c client.Client, subresource string, obj client.Object, opts ...client.SubResourceUpdateOption) error {
			if failStatus {
				return errors.New("status transport unavailable")
			}
			return c.SubResource(subresource).Update(ctx, obj, opts...)
		},
	})
	if err := r.reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	failStatus = false
	if err := r.reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	assertReadyConditionByName(t, r, "default", "route", "route", metav1.ConditionTrue, "Ready")
	route := &v1alpha1.IntelligentRoute{}
	key := client.ObjectKey{Namespace: "default", Name: "route"}
	if err := r.client.Get(context.Background(), key, route); err != nil {
		t.Fatal(err)
	}
	route.Status.Conditions = nil
	if err := r.client.Status().Update(context.Background(), route); err != nil {
		t.Fatal(err)
	}
	if err := r.reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	assertReadyConditionByName(t, r, "default", "route", "route", metav1.ConditionTrue, "Ready")
	if calls != 1 {
		t.Fatalf("status repair rebuilt generation %d times", calls)
	}
}

func TestReconcileReplacementAtSameGenerationActivates(t *testing.T) {
	pool := buildConflictPool("default", "pool")
	pool.UID = types.UID("old-pool")
	route := buildConflictRoute("default", "route")
	r := buildConflictReconciler(t, "default", pool, route)
	calls := 0
	r.onConfigUpdate = func(context.Context, *config.RouterConfig) error { calls++; return nil }
	if err := r.reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := r.client.Delete(context.Background(), pool); err != nil {
		t.Fatal(err)
	}
	replacement := buildConflictPool("default", "pool")
	replacement.UID = types.UID("new-pool")
	if err := r.client.Create(context.Background(), replacement); err != nil {
		t.Fatal(err)
	}
	if err := r.reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("replacement activation calls = %d", calls)
	}
}

func TestReconcileConflictRecoveryRepairsActiveGenerationStatus(t *testing.T) {
	r := buildConflictReconciler(t, "default", buildConflictPool("default", "pool"), buildConflictRoute("default", "route"))
	calls := 0
	r.onConfigUpdate = func(context.Context, *config.RouterConfig) error { calls++; return nil }
	ctx := context.Background()
	if err := r.reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	extra := buildConflictRoute("default", "extra")
	if err := r.client.Create(ctx, extra); err != nil {
		t.Fatal(err)
	}
	if err := r.reconcile(ctx); err == nil {
		t.Fatal("conflict accepted")
	}
	if err := r.client.Delete(ctx, extra); err != nil {
		t.Fatal(err)
	}
	if err := r.reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	assertReadyConditionByName(t, r, "default", "route", "route", metav1.ConditionTrue, "Ready")
	if calls != 1 {
		t.Fatalf("unchanged generation rebuilt %d times", calls)
	}
}

type activationTestCache struct{ cache.Cache }

func (activationTestCache) WaitForCacheSync(context.Context) bool { return true }

type activationTestManager struct {
	manager.Manager
	entered <-chan struct{}
}

func (m activationTestManager) GetCache() cache.Cache { return activationTestCache{} }
func (m activationTestManager) Start(ctx context.Context) error {
	select {
	case <-m.entered:
		return errors.New("terminal watcher failure")
	case <-ctx.Done():
		return ctx.Err()
	}
}

func TestReconcilerTerminalWatcherFailureCancelsActivation(t *testing.T) {
	r := buildConflictReconciler(t, "default", buildConflictPool("default", "pool"), buildConflictRoute("default", "route"))
	entered := make(chan struct{})
	r.runtimeManager = activationTestManager{entered: entered}
	r.onConfigUpdate = func(ctx context.Context, _ *config.RouterConfig) error {
		close(entered)
		<-ctx.Done()
		return ctx.Err()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	err := r.Start(ctx)
	if err == nil || !strings.Contains(err.Error(), "terminal watcher failure") {
		t.Fatalf("watcher failure = %v", err)
	}
	if ctx.Err() != nil {
		t.Fatal("watcher failure did not cancel pending activation")
	}
}
