//go:build !windows

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// GET /api/v1/inventory/classifier dumps the live runtime config. Resolved upstream
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

	req := httptest.NewRequest(http.MethodGet, "/api/v1/inventory/classifier", nil)
	rr := httptest.NewRecorder()
	s.handleClassifierInfo(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
	if strings.Contains(rr.Body.String(), canary) {
		t.Fatalf("/api/v1/inventory/classifier leaked a resolved credential value in its response body")
	}
}

// The loader keeps a password such as $Secr3tPass as written, since only an
// uppercase name is an environment reference. It must not pass as one here.
func TestClassifierInfoRedactsADollarPasswordThatIsNoReference(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.5
global:
  services:
    response_api:
      store_backend: redis
      redis:
        address: 127.0.0.1:6379
        password: "$Secr3tPass"
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	if cfg.ResponseAPI.Redis.Password != "$Secr3tPass" {
		t.Fatalf("loaded password = %q", cfg.ResponseAPI.Redis.Password)
	}
	s := &ClassificationAPIServer{config: cfg}

	rr := httptest.NewRecorder()
	s.handleClassifierInfo(rr, httptest.NewRequest(http.MethodGet, "/api/v1/inventory/classifier", nil))

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
	if strings.Contains(rr.Body.String(), "Secr3tPass") {
		t.Fatalf("/api/v1/inventory/classifier showed a password that is not an environment reference")
	}
}
