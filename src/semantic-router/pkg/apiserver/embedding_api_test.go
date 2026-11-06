//go:build !windows

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestEmbeddingAPIUnavailableReleasesSimilarityLease(t *testing.T) {
	for _, batch := range []bool{false, true} {
		cfg := &config.RouterConfig{}
		cfg.API.Embeddings.Enabled = true
		service := services.NewClassificationService(nil, cfg)
		api := &ClassificationAPIServer{classificationSvc: service, config: cfg}
		recorder := httptest.NewRecorder()
		if batch {
			api.handleBatchSimilarity(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"query":"hello","candidates":["hi"]}`)))
		} else {
			api.handleSimilarity(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"text1":"hello","text2":"hi"}`)))
		}
		if recorder.Code != http.StatusServiceUnavailable {
			t.Fatalf("unavailable runtime HTTP %d: %s", recorder.Code, recorder.Body.String())
		}
		closed := make(chan error, 1)
		go func() { closed <- service.Close() }()
		select {
		case err := <-closed:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(time.Second):
			t.Fatal("error response leaked a runtime lease")
		}
	}
}
