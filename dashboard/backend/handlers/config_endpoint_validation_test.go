package handlers

import (
	"strings"
	"testing"

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

// A backend endpoint is written as a deployment's is: host:port with the
// backend's protocol, or a full URL with its path.
func TestBackendEndpointTakesHostPortOrAFullURL(t *testing.T) {
	for _, endpoint := range []string{"127.0.0.1:8000", "vllm.example.com:8000", "http://upstream:18080/v1", "https://gateway.example.com/anthropic"} {
		if err := validateCanonicalEndpointRefs(backendEndpointConfig(endpoint)); err != nil {
			t.Errorf("%s: %v", endpoint, err)
		}
	}
	for endpoint, want := range map[string]string{
		"upstream:18080/v1":                   "write a full URL",
		"ftp://upstream/v1":                   "http or https",
		"https://user:secret@gateway.example": "credentials",
	} {
		err := validateCanonicalEndpointRefs(backendEndpointConfig(endpoint))
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: got %v, want an error about %q", endpoint, err, want)
		}
	}
}
