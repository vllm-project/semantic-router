package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRetrieveExternalRAGWindowsRanksByBestScore(t *testing.T) {
	tests := []struct {
		name      string
		format    string
		responses []map[string]interface{}
	}{
		{
			name:   "pinecone similarity",
			format: "pinecone",
			responses: []map[string]interface{}{
				pineconeResponse(pineconeMatch("first", 0.1), pineconeMatch("shared", 0.2)),
				pineconeResponse(pineconeMatch("best", 0.9), pineconeMatch("shared", 0.8)),
				pineconeResponse(pineconeMatch("third", 0.7)),
			},
		},
		{
			name:   "weaviate distance",
			format: "weaviate",
			responses: []map[string]interface{}{
				weaviateResponse(weaviateDocument("first", 0.9), weaviateDocument("shared", 0.8)),
				weaviateResponse(weaviateDocument("best", 0.1), weaviateDocument("shared", 0.2)),
				weaviateResponse(weaviateDocument("third", 0.3)),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var requestIndex atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				index := int(requestIndex.Add(1)) - 1
				if index >= len(tt.responses) {
					t.Errorf("received unexpected request %d", index+1)
					http.Error(w, "unexpected request", http.StatusInternalServerError)
					return
				}
				response := tt.responses[index]
				if err := json.NewEncoder(w).Encode(response); err != nil {
					t.Errorf("encode response: %v", err)
				}
			}))
			defer server.Close()

			requestBodies := make([][]byte, len(tt.responses))
			for i := range requestBodies {
				requestBodies[i] = []byte(`{}`)
			}
			documents, _, err := (&OpenAIRouter{}).retrieveExternalRAGWindows(
				context.Background(),
				&config.ExternalAPIRAGConfig{Endpoint: server.URL, RequestFormat: tt.format},
				requestBodies,
				3,
			)
			if err != nil {
				t.Fatalf("retrieveExternalRAGWindows() error = %v", err)
			}
			if got := int(requestIndex.Load()); got != len(tt.responses) {
				t.Fatalf("sent %d requests, want %d", got, len(tt.responses))
			}
			want := []string{"best", "shared", "third"}
			if !reflect.DeepEqual(documents, want) {
				t.Fatalf("documents = %v, want %v", documents, want)
			}
		})
	}
}

func pineconeResponse(matches ...map[string]interface{}) map[string]interface{} {
	values := make([]interface{}, len(matches))
	for i, match := range matches {
		values[i] = match
	}
	return map[string]interface{}{"matches": values}
}

func pineconeMatch(content string, score float64) map[string]interface{} {
	return map[string]interface{}{
		"metadata": map[string]interface{}{"content": content},
		"score":    score,
	}
}

func weaviateResponse(documents ...map[string]interface{}) map[string]interface{} {
	values := make([]interface{}, len(documents))
	for i, document := range documents {
		values[i] = document
	}
	return map[string]interface{}{
		"data": map[string]interface{}{
			"Get": map[string]interface{}{"Document": values},
		},
	}
}

func weaviateDocument(content string, distance float64) map[string]interface{} {
	return map[string]interface{}{
		"content":     content,
		"_additional": map[string]interface{}{"distance": distance},
	}
}

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
