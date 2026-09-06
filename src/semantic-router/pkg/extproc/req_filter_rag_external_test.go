package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRetrieveFromExternalAPIResponseLimit(t *testing.T) {
	responseBody, err := json.Marshal(map[string]interface{}{"content": "retrieved context"})
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(responseBody)
	}))
	defer server.Close()

	tests := []struct {
		name             string
		maxResponseBytes int64
		wantErr          bool
	}{
		{name: "default"},
		{name: "at limit", maxResponseBytes: int64(len(responseBody))},
		{name: "one byte over", maxResponseBytes: int64(len(responseBody)) - 1, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ragConfig := &config.RAGPluginConfig{
				Enabled: true,
				Backend: "external_api",
				BackendConfig: config.MustStructuredPayload(&config.ExternalAPIRAGConfig{
					Endpoint:         server.URL,
					RequestFormat:    "custom",
					RequestTemplate:  `{"query":"{{.Query}}"}`,
					MaxResponseBytes: tt.maxResponseBytes,
				}),
			}

			contextText, err := (&OpenAIRouter{}).retrieveFromExternalAPI(
				context.Background(),
				&RequestContext{UserContent: "hello"},
				ragConfig,
			)
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), "response body exceeds limit") {
					t.Fatalf("retrieveFromExternalAPI() error = %v, want response limit error", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("retrieveFromExternalAPI() error = %v", err)
			}
			if contextText != "retrieved context" {
				t.Fatalf("context = %q, want retrieved context", contextText)
			}
		})
	}
}
