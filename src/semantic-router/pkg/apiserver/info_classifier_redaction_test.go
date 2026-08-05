//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// GET /info/classifier dumps the live runtime config. Resolved upstream
// credentials (api_key_env expanded into VLLMEndpoint.APIKey /
// ModelParams.AccessKey) must never appear in that JSON response.
func TestClassifierInfoRedactsResolvedSecrets(t *testing.T) {
	// Sentinel stand-in for a resolved credential value. Deliberately not in the
	// gosec G101 credential-name/shape so the test itself does not trip the
	// hardcoded-credential linter.
	const canary = "redact-me-canary-value-0001"
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			VLLMEndpoints: []config.VLLMEndpoint{
				{Address: "10.0.0.1", Port: 8000, APIKey: canary},
			},
			ModelConfig: map[string]config.ModelParams{
				"m1": {AccessKey: canary},
			},
		},
	}
	s := &ClassificationAPIServer{config: cfg}

	req := httptest.NewRequest(http.MethodGet, "/info/classifier", nil)
	rr := httptest.NewRecorder()
	s.handleClassifierInfo(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
	if strings.Contains(rr.Body.String(), canary) {
		t.Fatalf("/info/classifier leaked a resolved credential value in its response body")
	}
}
