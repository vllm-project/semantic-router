package modelservice

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
)

func TestPhysicalInferenceExchangesShareBudgetAcrossSurfaces(t *testing.T) {
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()
	client, err := NewClient(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	ctx, ledger := budget.WithLimit(context.Background(), 2)
	for i, path := range []string{"/health", "/v1/bundle", "/v1/systemone", "/v1/systemone"} {
		method := http.MethodPost
		if i == 0 {
			method = http.MethodGet
		}
		req, err := http.NewRequestWithContext(ctx, method, server.URL+path, strings.NewReader(`{}`))
		if err != nil {
			t.Fatal(err)
		}
		response, err := client.httpClient.Do(req)
		if i == 3 {
			if !errors.Is(err, budget.ErrExhausted) || response != nil {
				t.Fatalf("expected exhaustion, response=%v error=%v", response, err)
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		_ = response.Body.Close()
	}
	if calls.Load() != 3 || ledger.Used() != 2 {
		t.Fatalf("physical exchanges=%d inference ledger=%d", calls.Load(), ledger.Used())
	}
}
