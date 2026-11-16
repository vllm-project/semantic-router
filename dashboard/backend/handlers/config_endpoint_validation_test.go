package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func backendEndpointConfig(endpoint string) routerconfig.CanonicalConfig {
	var configData routerconfig.CanonicalConfig
	configData.Providers.Models = []routerconfig.CanonicalProviderModel{{
		Name:        "general",
		BackendRefs: []routerconfig.CanonicalBackendRef{{Name: "primary", Endpoint: endpoint}},
	}}
	return configData
}

// A backend endpoint is read as the Router reads it: host:port, path optional,
// with the backend's protocol, or a full URL with its path.
func TestBackendEndpointTakesHostPortOrAFullURL(t *testing.T) {
	for _, endpoint := range []string{"127.0.0.1:8000", "vllm.example.com:8000", "upstream:18080/v1", "http://upstream:18080/v1", "https://gateway.example.com/anthropic"} {
		if err := validateCanonicalEndpointRefs(backendEndpointConfig(endpoint)); err != nil {
			t.Errorf("%s: %v", endpoint, err)
		}
	}
	for endpoint, want := range map[string]string{
		"ftp://upstream/v1":                   "http or https",
		"https://user:secret@gateway.example": "credentials",
	} {
		err := validateCanonicalEndpointRefs(backendEndpointConfig(endpoint))
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: got %v, want an error about %q", endpoint, err, want)
		}
	}
}

// The setup wizard sends a vLLM base URL such as http://localhost:8000/v1 as
// endpoint localhost:8000/v1 with protocol http, which the Router loads.
func TestSetupValidateAcceptsTheWizardEndpointWithPath(t *testing.T) {
	isolateConfigMutationRuntime(t)
	configPath := createBootstrapSetupConfig(t, t.TempDir())
	body := `{"config":{"providers":{"models":[{"name":"m","provider_model_id":"m","backend_refs":[{"name":"primary","weight":100,"endpoint":"localhost:8000/v1","protocol":"http","provider":"vllm"}]}],"defaults":{"model":"m"}},"recipes":[{"name":"default","decisions":[{"name":"default-route","priority":100,"rules":{"operator":"AND","conditions":[]},"modelRefs":[{"model":"m","use_reasoning":false}]}]}]}}`
	w := httptest.NewRecorder()
	SetupValidateHandler(configPath, setupmode.New(configPath, false))(w, httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewBufferString(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("setup validate = %d %s, want 200", w.Code, strings.TrimSpace(w.Body.String()))
	}
	var resp struct {
		Valid  bool `json:"valid"`
		Config struct {
			Providers struct {
				Models []struct {
					BackendRefs []struct{ Endpoint, Protocol string } `json:"backend_refs"`
				} `json:"models"`
			} `json:"providers"`
		} `json:"config"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil || len(resp.Config.Providers.Models) != 1 {
		t.Fatalf("decode response %s: %v", w.Body.String(), err)
	}
	ref := resp.Config.Providers.Models[0].BackendRefs[0]
	if !resp.Valid || ref.Endpoint != "localhost:8000/v1" || ref.Protocol != "http" {
		t.Fatalf("valid=%v endpoint=%q protocol=%q, want true localhost:8000/v1 http", resp.Valid, ref.Endpoint, ref.Protocol)
	}
}
