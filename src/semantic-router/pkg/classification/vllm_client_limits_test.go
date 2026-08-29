package classification

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func newLimitedVLLMClient(server *httptest.Server, maxResponseBytes int64) *VLLMClient {
	client := NewVLLMClient(&config.ExternalModelConfig{
		ModelEndpoint:    config.ClassifierVLLMEndpoint{Address: "placeholder", Port: 1},
		MaxResponseBytes: maxResponseBytes,
	})
	client.baseURL = server.URL
	return client
}

func TestVLLMClientResponseOneByteOverLimitIsRejected(t *testing.T) {
	const limit = 1024
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(strings.Repeat("x", limit+1)))
	}))
	defer server.Close()

	client := newLimitedVLLMClient(server, limit)
	_, err := client.Generate(context.Background(), "classifier", "test", nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "exceeds limit") {
		t.Fatalf("error = %v, want exceeded limit", err)
	}
}

func TestVLLMClientErrorBodyIsTruncated(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(strings.Repeat("e", int(maxClassifyErrorBodyBytes)+1)))
	}))
	defer server.Close()

	client := newLimitedVLLMClient(server, 1)
	_, err := client.Generate(context.Background(), "classifier", "test", nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "status 502") || !strings.Contains(err.Error(), "truncated=true") {
		t.Fatalf("error = %v, want status and truncation", err)
	}
	if len(err.Error()) > int(maxClassifyErrorBodyBytes)+256 {
		t.Fatalf("error length = %d, want bounded error", len(err.Error()))
	}
}
