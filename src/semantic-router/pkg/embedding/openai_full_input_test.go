package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOpenAICompatibleFullInputPreservesRequestAndLimitErrors(t *testing.T) {
	long := strings.Repeat("complete input ", 10000) + "distinct ending"
	seen := []string{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		if len(request.Input) != 1 {
			t.Errorf("inputs=%d", len(request.Input))
			return
		}
		seen = append(seen, request.Input[0])
		if request.Input[0] == long {
			http.Error(w, "maximum context length exceeded", http.StatusBadRequest)
			return
		}
		writeEmbeddingResponse(t, w, [][]float64{{1, 0}})
	}))
	defer server.Close()
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{BaseURL: server.URL, Model: "embedding-model"})
	for _, candidate := range []Provider{provider, WithOptions(provider, Options{})} {
		vector, err := EmbedFullInput(context.Background(), candidate, "short complete query", Options{})
		if err != nil || len(vector) != 2 {
			t.Fatalf("valid input=%v %v", vector, err)
		}
		if _, err := EmbedFullInput(context.Background(), candidate, long, Options{}); err == nil || !strings.Contains(err.Error(), "HTTP status 400") {
			t.Fatalf("overlimit input=%v", err)
		}
	}
	if len(seen) != 4 || seen[1] != long || seen[3] != long {
		t.Fatal("endpoint did not receive both complete oversized inputs")
	}
}
