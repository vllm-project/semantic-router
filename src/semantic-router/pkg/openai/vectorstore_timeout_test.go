package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestVectorStoreClientTimeoutHonorsConfig(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(2 * time.Second)
		_, _ = w.Write([]byte(`{"data":[]}`))
	}))
	defer server.Close()

	client := NewVectorStoreClientWithTimeout(server.URL, "test-key", 0, time.Second)

	start := time.Now()
	_, err := client.SearchVectorStore(context.Background(), "vs_1", "query", 1, nil)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected a timeout error, got nil")
	}
	if elapsed > 1500*time.Millisecond {
		t.Fatalf("elapsed = %v, want < 1.5s; client timeout did not fire", elapsed)
	}
}
