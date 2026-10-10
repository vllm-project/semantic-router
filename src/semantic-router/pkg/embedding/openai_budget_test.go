package embedding

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
)

func TestRemoteEmbeddingBudgetCoversBothTransports(t *testing.T) {
	for _, injected := range []bool{false, true} {
		t.Run(map[bool]string{false: "connector", true: "injected"}[injected], func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if calls.Add(1) == 1 {
					w.WriteHeader(http.StatusServiceUnavailable)
					return
				}
				_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[1,0]}]}`))
			}))
			defer server.Close()
			cfg := OpenAICompatibleConfig{BaseURL: server.URL, Model: "embedding", MaxRetries: 3}
			if injected {
				cfg.HTTPClient = server.Client()
			}
			provider, err := NewOpenAICompatibleProvider(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer provider.Close()
			ctx, ledger := budget.WithLimit(t.Context(), 1)
			_, err = provider.Embed(ctx, "first")
			if !errors.Is(err, budget.ErrExhausted) || calls.Load() != 1 || ledger.Used() != 1 {
				t.Fatalf("calls=%d ledger=%d error=%v", calls.Load(), ledger.Used(), err)
			}
			ctx, ledger = budget.WithLimit(t.Context(), 1)
			_, err = provider.Embed(ctx, "second")
			if err != nil || calls.Load() != 2 || ledger.Used() != 1 {
				t.Fatalf("successful exchange charged twice: calls=%d ledger=%d error=%v", calls.Load(), ledger.Used(), err)
			}
		})
	}
}
