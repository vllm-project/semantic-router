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

// Xunzhuo's blocker on #2507: load-time template validation is worthless if the
// runtime path does not use the same renderer. The old buildCustomRequest
// string-replaced user content into the template text, so a quote or backslash
// could close a JSON string early and reshape the request while validation
// still passed. These cases send exactly that through the real runtime path.
func TestBuildCustomRequestEscapesHostileUserContent(t *testing.T) {
	topK := 3
	threshold := float32(0.42)
	ragConfig := &config.RAGPluginConfig{
		TopK:                &topK,
		SimilarityThreshold: &threshold,
	}
	const template = `{"query":"${user_content}","top_k":${top_k},"threshold":${threshold}}`

	for _, tc := range []struct {
		name  string
		input string
	}{
		{"double quote", `say "hello"`},
		{"quote then injected key", `x","top_k":999,"evil":"`},
		{"trailing backslash", `path\`},
		{"escaped quote", `a\"b`},
		{"newline and tab", "line1\nline2\tend"},
		{"brace and dollar markers", `${user_content} {{.Query}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := &OpenAIRouter{}
			body, err := router.buildCustomRequest(
				&RequestContext{UserContent: tc.input}, ragConfig, template)
			if err != nil {
				t.Fatalf("buildCustomRequest() error = %v", err)
			}

			// The request must still be a JSON object with exactly the shape
			// the template declared -- no injected keys, no truncation.
			var decoded map[string]interface{}
			if err := json.Unmarshal(body, &decoded); err != nil {
				t.Fatalf("rendered body is not valid JSON: %v\nbody: %s", err, body)
			}
			if len(decoded) != 3 {
				t.Fatalf("expected exactly 3 keys, got %d: %v", len(decoded), decoded)
			}
			if got, ok := decoded["query"].(string); !ok || got != tc.input {
				t.Fatalf("query round-trip failed: got %#v, want %#v", decoded["query"], tc.input)
			}
			if got, ok := decoded["top_k"].(float64); !ok || int(got) != topK {
				t.Fatalf("top_k = %#v, want %d", decoded["top_k"], topK)
			}
		})
	}
}
