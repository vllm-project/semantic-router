package openai

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// The base URL is configured by the operator, so a redirect must not replay
// the API key, or the file itself, to the origin it names.
func TestStoreClientsRefuseRedirects(t *testing.T) {
	followed := 0
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		followed++
		if r.Header.Get("Authorization") != "" {
			t.Error("the redirect target received the API key")
		}
	}))
	defer target.Close()
	source := httptest.NewServer(http.RedirectHandler(target.URL, http.StatusFound))
	defer source.Close()

	clients := map[string]*http.Client{
		"vector store": NewVectorStoreClientWithTimeout(source.URL, "key", 0, time.Second).httpClient,
		"file store":   NewFileStoreClient(source.URL, "key").httpClient,
	}
	for name, client := range clients {
		request, _ := http.NewRequest(http.MethodGet, source.URL+"/v1/vector_stores", nil)
		request.Header.Set("Authorization", "Bearer key")
		response, err := client.Do(request)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		response.Body.Close()
		if response.StatusCode != http.StatusFound {
			t.Errorf("%s: status = %d, want the redirect itself", name, response.StatusCode)
		}
	}
	if followed != 0 {
		t.Fatalf("the redirect target was called %d times", followed)
	}
}
