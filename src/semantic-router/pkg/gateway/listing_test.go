package gateway

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// A keyed listener that serves Chat and systemone answers GET /v1/models with
// both listings: the systemone handler checks the listener's key itself, so
// the credential must reach it.
func TestKeyedListenerListsSystemOneModels(t *testing.T) {
	p := &fakeProcessor{effects: map[routing.Phase][]*routing.Effect{
		routing.PhaseRequestHeaders: {{Immediate: &routing.ImmediateResponse{
			Status: http.StatusOK,
			Header: &routing.HeaderMutation{Set: []routing.HeaderOption{set("content-type", "application/json")}},
			Body:   []byte(`{"object":"list","data":[{"id":"llm-a"}]}`),
		}}},
	}}
	handler, err := NewHandler(Options{Serving: Static(Serving{
		Engine:         routing.NewEngine(p, routing.DefaultOptions),
		Upstream:       &fakeUpstream{resp: okUpstream("unused", http.Header{})},
		APIKeys:        []string{"key"},
		Models:         []string{"llm-a"},
		ListsLLM:       true,
		ListsSystemOne: true,
		SystemOne: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !newAPIKeys([]string{"key"}).check(r) {
				writeUnauthorized(w)
				return
			}
			_, _ = w.Write([]byte(`{"models":[{"name":"systemone/auto"}]}`))
		}),
	})})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	request.Header.Set("Authorization", "Bearer key")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	body := response.Body.String()
	if response.Code != http.StatusOK || !strings.Contains(body, "llm-a") || !strings.Contains(body, "systemone/auto") {
		t.Fatalf("listing = %d %s", response.Code, body)
	}

	denied := httptest.NewRecorder()
	handler.ServeHTTP(denied, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if denied.Code != http.StatusUnauthorized {
		t.Fatalf("no key = %d", denied.Code)
	}
}
