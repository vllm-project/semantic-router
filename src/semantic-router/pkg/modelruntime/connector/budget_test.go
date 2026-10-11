package connector

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
)

func TestInferenceBudgetCountsPhysicalRetries(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()
	options := testOptions()
	options.MaxRetries = 4
	client := newTestClient(t, server, options)
	ctx, ledger := budget.WithLimit(t.Context(), 2)
	_, err := client.Do(ctx, testOperation, []byte(`{}`))
	if !errors.Is(err, budget.ErrExhausted) || ledger.Used() != 2 || calls.Load() != 2 {
		t.Fatalf("calls=%d ledger=%d error=%v", calls.Load(), ledger.Used(), err)
	}
	_ = assertConnectorError(t, err, KindBudget, 3, false)
}

func TestDiscoveryDoesNotSpendInferenceBudget(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()
	client := newTestClient(t, server, testOptions())
	ctx, ledger := budget.WithLimit(t.Context(), 0)
	_, err := client.Do(ctx, Operation{Name: "models", Method: http.MethodGet}, nil)
	if err != nil || ledger.Used() != 0 || calls.Load() != 1 {
		t.Fatalf("calls=%d ledger=%d error=%v", calls.Load(), ledger.Used(), err)
	}
	_, err = client.Do(ctx, testOperation, nil)
	if !errors.Is(err, budget.ErrExhausted) || calls.Load() != 1 {
		t.Fatalf("zero-budget inference reached backend: calls=%d error=%v", calls.Load(), err)
	}
}

func TestCanceledInferenceDoesNotSpendBudget(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("canceled request reached backend")
	}))
	defer server.Close()
	client := newTestClient(t, server, testOptions())
	ctx, cancel := context.WithCancel(t.Context())
	ctx, ledger := budget.WithLimit(ctx, 2)
	cancel()
	_, err := client.Do(ctx, testOperation, nil)
	if !errors.Is(err, context.Canceled) || ledger.Used() != 0 {
		t.Fatalf("ledger=%d error=%v", ledger.Used(), err)
	}
}
