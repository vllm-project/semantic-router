package vectorstore

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// The endpoint is configured by the operator, so a redirect must not replay
// the auth token to the origin it names.
func TestLlamaStackBackendRefusesRedirects(t *testing.T) {
	followed := 0
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		followed++
		if r.Header.Get("Authorization") != "" {
			t.Error("the redirect target received the auth token")
		}
	}))
	defer target.Close()
	source := httptest.NewServer(http.RedirectHandler(target.URL, http.StatusFound))
	defer source.Close()

	backend, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: source.URL, AuthToken: "token"})
	if err != nil {
		t.Fatal(err)
	}
	request, _ := http.NewRequest(http.MethodGet, source.URL+"/v1/vector_stores", nil)
	request.Header.Set("Authorization", "Bearer token")
	response, err := backend.httpClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if followed != 0 {
		t.Fatalf("the redirect target was called %d times", followed)
	}
	if response.StatusCode != http.StatusFound {
		t.Fatalf("status = %d, want the redirect itself", response.StatusCode)
	}
}
