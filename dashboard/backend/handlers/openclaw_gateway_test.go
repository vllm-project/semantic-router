package handlers

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultOpenClawModelBaseURL(t *testing.T) {
	t.Setenv("OPENCLAW_MODEL_BASE_URL", "")
	t.Setenv("TARGET_ENVOY_URL", "")
	if got := defaultOpenClawModelBaseURL(); got != "http://127.0.0.1:8801/v1" {
		t.Fatalf("expected fallback model base URL, got %q", got)
	}

	t.Setenv("OPENCLAW_MODEL_BASE_URL", "http://localhost:9999/v1")
	if got := defaultOpenClawModelBaseURL(); got != "http://localhost:9999/v1" {
		t.Fatalf("expected env model base URL, got %q", got)
	}
}

func TestDefaultOpenClawModelBaseURL_UsesTargetEnvoyURL(t *testing.T) {
	t.Setenv("OPENCLAW_MODEL_BASE_URL", "")
	t.Setenv("TARGET_ENVOY_URL", "http://vllm-sr-envoy-container:8899")

	if got := defaultOpenClawModelBaseURL(); got != "http://vllm-sr-envoy-container:8899/v1" {
		t.Fatalf("expected TARGET_ENVOY_URL-derived model base URL, got %q", got)
	}
}

func TestOpenClawModelGatewayContainerNamePrefersExplicitOverride(t *testing.T) {
	t.Setenv("OPENCLAW_MODEL_GATEWAY_CONTAINER_NAME", "explicit-model-gateway")
	t.Setenv("TARGET_ENVOY_URL", "http://vllm-sr-envoy-container:8899")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv("OPENCLAW_DASHBOARD_CONTAINER_NAME", "lane-a-vllm-sr-dashboard-container")

	if got := openClawModelGatewayContainerName(); got != "explicit-model-gateway" {
		t.Fatalf("openClawModelGatewayContainerName() = %q, want explicit-model-gateway", got)
	}
}

func TestOpenClawModelGatewayContainerNameDerivesFromTargetEnvoyURL(t *testing.T) {
	t.Setenv("OPENCLAW_MODEL_GATEWAY_CONTAINER_NAME", "")
	t.Setenv("TARGET_ENVOY_URL", "http://vllm-sr-envoy-container:8899")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv("OPENCLAW_DASHBOARD_CONTAINER_NAME", "lane-a-vllm-sr-dashboard-container")

	if got := openClawModelGatewayContainerName(); got != "vllm-sr-envoy-container" {
		t.Fatalf("openClawModelGatewayContainerName() = %q, want vllm-sr-envoy-container", got)
	}
}

func TestOpenClawModelGatewayContainerNameFallsBackToManagedEnvoyContainer(t *testing.T) {
	t.Setenv("OPENCLAW_MODEL_GATEWAY_CONTAINER_NAME", "")
	t.Setenv("TARGET_ENVOY_URL", "")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv("OPENCLAW_DASHBOARD_CONTAINER_NAME", "lane-a-vllm-sr-dashboard-container")

	if got := openClawModelGatewayContainerName(); got != "lane-a-vllm-sr-envoy-container" {
		t.Fatalf("openClawModelGatewayContainerName() = %q, want lane-a-vllm-sr-envoy-container", got)
	}
}

func TestResolveOpenClawModelBaseURL_TargetEnvoyWinsOverRouterConfig(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	configYAML := `
listeners:
  - address: 0.0.0.0
    port: 18889
`
	if err := os.WriteFile(configPath, []byte(configYAML), 0o644); err != nil {
		t.Fatalf("failed to write config file: %v", err)
	}

	h := NewOpenClawHandler(tempDir, false)
	h.SetRouterConfigPath(configPath)
	t.Setenv("OPENCLAW_MODEL_BASE_URL", "")
	t.Setenv("TARGET_ENVOY_URL", "http://vllm-sr-envoy-container:8899")

	if got := h.resolveOpenClawModelBaseURL(); got != "http://vllm-sr-envoy-container:8899/v1" {
		t.Fatalf("expected TARGET_ENVOY_URL model base URL, got %q", got)
	}
}
