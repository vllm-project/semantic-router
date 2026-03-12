package handlers

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteOpenClawConfig_UsesConfiguredModelContextWindow(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "openclaw.json")

	req := ProvisionRequest{
		Container: ContainerConfig{
			GatewayPort:        18788,
			AuthToken:          "test-token",
			ModelBaseURL:       "http://localhost:8080",
			ModelAPIKey:        "test-api-key",
			ModelName:          "auto",
			ModelContextWindow: 300000,
			MemoryBackend:      "local",
		},
	}

	if err := writeOpenClawConfig(configPath, req); err != nil {
		t.Fatalf("writeOpenClawConfig failed: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read config file: %v", err)
	}

	var cfg map[string]interface{}
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("config file should be valid JSON: %v", err)
	}

	modelsCfg, ok := cfg["models"].(map[string]interface{})
	if !ok {
		t.Fatalf("models block missing or invalid")
	}
	providersCfg, ok := modelsCfg["providers"].(map[string]interface{})
	if !ok {
		t.Fatalf("models.providers block missing or invalid")
	}
	vllmCfg, ok := providersCfg["vllm"].(map[string]interface{})
	if !ok {
		t.Fatalf("models.providers.vllm block missing or invalid")
	}
	models, ok := vllmCfg["models"].([]interface{})
	if !ok || len(models) != 1 {
		t.Fatalf("vllm.models should contain one entry, got %#v", vllmCfg["models"])
	}
	modelEntry, ok := models[0].(map[string]interface{})
	if !ok {
		t.Fatalf("vllm.models[0] block missing or invalid")
	}
	if got := modelEntry["contextWindow"]; got != float64(300000) {
		t.Fatalf("contextWindow mismatch: got %v", got)
	}
	if _, exists := modelEntry["maxTokens"]; exists {
		t.Fatalf("maxTokens should be omitted so OpenClaw can use its provider default")
	}
	compatCfg, ok := modelEntry["compat"].(map[string]interface{})
	if !ok {
		t.Fatalf("compat block missing or invalid")
	}
	if got := compatCfg["maxTokensField"]; got != "max_tokens" {
		t.Fatalf("compat.maxTokensField mismatch: got %v", got)
	}
}

func TestDefaultOpenClawModelContextWindow_UsesEnvOverride(t *testing.T) {
	t.Setenv("OPENCLAW_MODEL_CONTEXT_WINDOW", "300000")

	if got := defaultOpenClawModelContextWindow(); got != 300000 {
		t.Fatalf("defaultOpenClawModelContextWindow() = %d, want 300000", got)
	}
}

func TestDefaultOpenClawModelContextWindow_UsesFallback(t *testing.T) {
	t.Setenv("OPENCLAW_MODEL_CONTEXT_WINDOW", "")

	if got := defaultOpenClawModelContextWindow(); got != 262144 {
		t.Fatalf("defaultOpenClawModelContextWindow() = %d, want 262144", got)
	}
}
