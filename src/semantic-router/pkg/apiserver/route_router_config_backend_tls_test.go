package apiserver

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Envoy reads a backend's CA, server name and client certificate when it
// starts, so with --gateway extproc a reload that changes them takes the
// deployment workflow, as Envoy and the Router must keep trusting the same
// CA. Standalone mode rebuilds its connection pools.
func TestReloadOfABackendTLSChangeFollowsTheGatewayMode(t *testing.T) {
	withCA := func(caFile string) *config.RouterConfig {
		cfg := minimalDeployTestConfig("route")
		cfg.VLLMEndpoints[0].ProviderProfileName = "primary"
		cfg.VLLMEndpoints[0].Protocol = "https"
		cfg.ProviderProfiles = map[string]config.ProviderProfile{"primary": {
			Type: "vllm", BaseURL: "https://vllm.internal.example:8000",
			TLS: &config.DeploymentTLS{CAFile: caFile, ServerName: "vllm.internal.example"},
		}}
		return cfg
	}
	current := withCA("/etc/vllm-sr/certs/example-corp-ca.pem")
	rotated := withCA("/etc/vllm-sr/certs/example-corp-ca-2027.pem")

	err := validateParsedHotReloadCompatibilityInMode(config.GatewayExtProc, current, rotated)
	if err == nil || !strings.Contains(err.Error(), "rendered into Envoy") {
		t.Fatalf("Envoy renders a backend's CA in extproc mode, got %v", err)
	}
	if err := validateParsedHotReloadCompatibilityInMode(config.GatewayStandalone, current, rotated); err != nil {
		t.Fatalf("standalone mode reloads a backend's CA: %v", err)
	}
	unchanged := withCA("/etc/vllm-sr/certs/example-corp-ca.pem")
	if err := validateParsedHotReloadCompatibilityInMode(config.GatewayExtProc, current, unchanged); err != nil {
		t.Fatalf("an unchanged tls block reloads: %v", err)
	}
}
