package k8s

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Keep normal object operations in the fake client, but send the stalled status
// update through controller-runtime's real REST/HTTP transport.
func reconcilerWithStalledStatus(t *testing.T, kind string) (*Reconciler, <-chan struct{}, <-chan struct{}) {
	t.Helper()
	r := buildConflictReconciler(t, "default", buildConflictPool("default", "pool"), buildConflictRoute("default", "route"))
	entered, canceled := make(chan struct{}), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPut || !strings.HasSuffix(request.URL.Path, "/status") {
			t.Errorf("unexpected Kubernetes request: %s %s", request.Method, request.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if _, err := io.Copy(io.Discard, request.Body); err != nil {
			t.Errorf("read status request: %v", err)
			return
		}
		close(entered)
		<-request.Context().Done()
		close(canceled)
	}))
	t.Cleanup(server.Close)
	mapper := apimeta.NewDefaultRESTMapper([]schema.GroupVersion{v1alpha1.GroupVersion})
	mapper.Add(v1alpha1.GroupVersion.WithKind("IntelligentPool"), apimeta.RESTScopeNamespace)
	mapper.Add(v1alpha1.GroupVersion.WithKind("IntelligentRoute"), apimeta.RESTScopeNamespace)
	transport, err := client.New(&rest.Config{Host: server.URL}, client.Options{Scheme: r.scheme, Mapper: mapper})
	if err != nil {
		t.Fatal(err)
	}
	var stalled atomic.Bool
	r.client = interceptor.NewClient(r.client.(client.WithWatch), interceptor.Funcs{
		SubResourceUpdate: func(ctx context.Context, underlying client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
			_, pool := object.(*v1alpha1.IntelligentPool)
			if (kind == "pool") == pool && stalled.CompareAndSwap(false, true) {
				return transport.SubResource(subresource).Update(ctx, object, opts...)
			}
			return underlying.SubResource(subresource).Update(ctx, object, opts...)
		},
	})
	return r, entered, canceled
}

func TestReconcileStatusTransportTimeoutAllowsActivation(t *testing.T) {
	for _, kind := range []string{"pool", "route"} {
		t.Run(kind, func(t *testing.T) {
			r, entered, transportCanceled := reconcilerWithStalledStatus(t, kind)
			parent, cancel := context.WithCancel(context.Background())
			activated := make(chan context.Context, 1)
			r.onConfigUpdate = func(ctx context.Context, _ *config.RouterConfig) error {
				activated <- ctx
				return ctx.Err()
			}
			finished := make(chan struct{})
			result := make(chan error, 1)
			go func() { defer close(finished); result <- r.reconcile(parent) }()
			defer func() { cancel(); <-finished }()
			select {
			case <-entered:
			case <-time.After(time.Second):
				t.Fatal("status request did not reach the HTTP transport")
			}
			select {
			case activationContext := <-activated:
				if activationContext != parent || activationContext.Err() != nil {
					t.Fatalf("activation inherited the expired status context: %v", activationContext.Err())
				}
			case <-time.After(3 * time.Second):
				t.Fatal("pending status transport prevented activation beyond the two-second request budget")
			}
			if err := <-result; err != nil {
				t.Fatal(err)
			}
			select {
			case <-transportCanceled:
			case <-time.After(time.Second):
				t.Fatal("status timeout did not cancel the actual HTTP request")
			}
			assertReadyConditionByName(t, r, "default", "pool", "pool", metav1.ConditionTrue, "Ready")
			assertReadyConditionByName(t, r, "default", "route", "route", metav1.ConditionTrue, "Ready")
		})
	}
}

func TestReconcileStatusTransportHonorsParentCancellation(t *testing.T) {
	r, entered, transportCanceled := reconcilerWithStalledStatus(t, "pool")
	parent, cancel := context.WithCancel(context.Background())
	r.onConfigUpdate = func(ctx context.Context, _ *config.RouterConfig) error {
		if ctx.Err() == nil {
			t.Error("activation context outlived the canceled parent")
		}
		return ctx.Err()
	}
	result := make(chan error, 1)
	finished := make(chan struct{})
	go func() { defer close(finished); result <- r.reconcile(parent) }()
	defer func() { cancel(); <-finished }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("status request did not reach the HTTP transport")
	}
	cancel()
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("reconcile after parent cancellation: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("parent cancellation waited for the status budget")
	}
	select {
	case <-transportCanceled:
	case <-time.After(time.Second):
		t.Fatal("parent cancellation did not cancel HTTP transport")
	}
}

func TestReconcileStatusErrorsRepairWithoutReactivation(t *testing.T) {
	r := buildConflictReconciler(t, "default", buildConflictPool("default", "pool"), buildConflictRoute("default", "route"))
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	calls, failStatus := 0, true
	var statusContexts []context.Context
	r.client = interceptor.NewClient(r.client.(client.WithWatch), interceptor.Funcs{
		SubResourceUpdate: func(ctx context.Context, underlying client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
			statusContexts = append(statusContexts, ctx)
			deadline, ok := ctx.Deadline()
			if !ok || time.Until(deadline) > 2*time.Second {
				t.Error("status request has no bounded child context")
			}
			if failStatus {
				return errors.New("status transport unavailable")
			}
			return underlying.SubResource(subresource).Update(ctx, object, opts...)
		},
	})
	r.onConfigUpdate = func(ctx context.Context, _ *config.RouterConfig) error {
		calls++
		if ctx != parent {
			t.Error("activation context was replaced")
		}
		return ctx.Err()
	}
	if err := r.reconcile(parent); err != nil {
		t.Fatal(err)
	}
	if parent.Err() != nil {
		t.Fatal("status failure canceled parent")
	}
	for _, ctx := range statusContexts {
		if !errors.Is(ctx.Err(), context.Canceled) {
			t.Error("completed status request retained its timer/context")
		}
	}
	failStatus = false
	if err := r.reconcile(parent); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("status-only repair reactivated runtime %d times", calls)
	}
	assertReadyConditionByName(t, r, "default", "pool", "pool", metav1.ConditionTrue, "Ready")
	assertReadyConditionByName(t, r, "default", "route", "route", metav1.ConditionTrue, "Ready")
}
