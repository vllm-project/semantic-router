package upstream

import (
	"context"
	"errors"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
)

func TestExactRouteNeverUsesDefaultChatBackend(t *testing.T) {
	var calls atomic.Int64
	server := backend(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusOK)
	})
	set := newSet(t, Options{}, clusterOf("chat", endpointOf(t, "chat", server)))
	req := post("missing-native-model")
	req.ExactRoute = true
	if response, err := set.Do(context.Background(), req); KindOf(err) != KindNoRoute || response != nil {
		t.Fatalf("unknown exact route: response=%v error=%v", response, err)
	}
	if calls.Load() != 0 {
		t.Fatal("native request reached the default Chat backend")
	}
}

func TestProviderRetriesShareRoutingCallBudget(t *testing.T) {
	var calls atomic.Int64
	server := backend(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	})
	set := newSet(t, Options{Clock: &waitRecorder{}}, clusterOf("native", endpointOf(t, "native", server)))
	ctx, ledger := budget.WithLimit(context.Background(), 3)
	// One signal exchange has already consumed part of this request's limit.
	if err := budget.Consume(ctx); err != nil {
		t.Fatal(err)
	}
	req := post("native")
	req.ExactRoute = true
	req.Policy = &Policy{Retry: &RetryPolicy{On: Retry5xx, NumRetries: 5}}
	response, err := set.Do(ctx, req)
	if response != nil || !errors.Is(err, budget.ErrExhausted) || KindOf(err) != KindBudgetExhausted {
		t.Fatalf("response=%v error=%v", response, err)
	}
	if calls.Load() != 2 || ledger.Used() != 3 {
		t.Fatalf("backend calls=%d ledger=%d", calls.Load(), ledger.Used())
	}
}
