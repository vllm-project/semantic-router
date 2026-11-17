package extproc

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestExternalAPIClientHonorsConfiguredTimeout(t *testing.T) {
	sixty := 60
	cfg := &config.ExternalAPIRAGConfig{TimeoutSeconds: &sixty}
	if got := externalAPIClient(cfg.GetTimeout()).Timeout; got != 60*time.Second {
		t.Fatalf("client timeout = %v, want 60s", got)
	}
	if externalAPIClient(time.Second).Transport != externalAPIClient(time.Minute).Transport {
		t.Fatal("clients must share one transport")
	}
}

// An endpoint that redirects must not see the configured credential headers
// or the query replayed: the client refuses the redirect instead.
func TestExternalAPIClientRefusesRedirects(t *testing.T) {
	followed := 0
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		followed++
		if r.Header.Get("x-api-key") != "" {
			t.Error("the redirect target received the configured credential")
		}
	}))
	defer target.Close()
	source := httptest.NewServer(http.RedirectHandler(target.URL, http.StatusFound))
	defer source.Close()

	request, _ := http.NewRequest(http.MethodPost, source.URL, strings.NewReader(`{"query":"secret"}`))
	request.Header.Set("x-api-key", "secret")
	response, err := externalAPIClient(time.Second).Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if followed != 0 {
		t.Fatalf("the redirect target was called %d times", followed)
	}
	if response.StatusCode != http.StatusFound {
		t.Fatalf("status = %d, want the redirect itself", response.StatusCode)
	}
}
