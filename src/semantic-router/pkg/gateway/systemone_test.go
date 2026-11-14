package gateway

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func TestNativeSystemOneBypassesChatPipeline(t *testing.T) {
	calls := 0
	handler, err := NewHandler(Options{Serving: Static(Serving{SystemOne: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; w.WriteHeader(202) })})})
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/v1/systemone", "/v1/decisions", "/v1/systemone/models"} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, path, nil))
		if response.Code != 202 {
			t.Fatalf("%s reached Chat", path)
		}
	}
	if calls != 3 {
		t.Fatal(calls)
	}
}

func TestEngineFrontendRejectsChatButKeepsNative(t *testing.T) {
	handler, err := NewHandler(Options{Serving: Static(Serving{RoutingDisabled: true, APIKeys: []string{"key"}, SystemOne: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusAccepted) })})})
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/v1/chat/completions", "/v1/responses", "/v1/models"} {
		response := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodPost, path, nil)
		request.Header.Set("Authorization", "Bearer key")
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusNotFound {
			t.Fatalf("%s = %d", path, response.Code)
		}
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/systemone", nil))
	if response.Code != http.StatusAccepted {
		t.Fatal("native surface stopped with routing")
	}
}

func TestListingKeepsTheCatalogCodexAskedFor(t *testing.T) {
	systemOne := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"models":[{"name":"vela3-lite"}]}`))
	})
	for path, want := range map[string]string{
		"/v1/models?client_version=0.166.0": `{"data":[],"models":[],"object":"list"}`,
		"/v1/models":                        `{"data":[],"models":[{"name":"vela3-lite"}],"object":"list"}`,
	} {
		body := `{"data":[],"object":"list"}`
		if strings.Contains(path, "client_version") {
			body = `{"data":[],"models":[],"object":"list"}`
		}
		p := &fakeProcessor{effects: map[routing.Phase][]*routing.Effect{
			routing.PhaseRequestHeaders: {{Immediate: &routing.ImmediateResponse{Status: 200, Body: []byte(body)}}},
		}}
		handler, err := NewHandler(Options{Serving: Static(Serving{
			Engine: routing.NewEngine(p, routing.DefaultOptions), Upstream: &fakeUpstream{},
			SystemOne: systemOne, ListsLLM: true, ListsSystemOne: true,
		})})
		if err != nil {
			t.Fatal(err)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, path, nil))
		if got := strings.TrimSpace(response.Body.String()); got != want {
			t.Fatalf("%s = %s, want %s", path, got, want)
		}
	}
}
