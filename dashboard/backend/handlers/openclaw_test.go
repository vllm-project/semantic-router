package handlers

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteOpenClawConfig_ReplacesDirectoryPath(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "openclaw.json")

	// Simulate stale broken state from a bad bind mount where openclaw.json
	// was created as a directory on host.
	if err := os.MkdirAll(configPath, 0o755); err != nil {
		t.Fatalf("failed to create stale config dir: %v", err)
	}

	req := ProvisionRequest{
		Container: ContainerConfig{
			GatewayPort:   18788,
			AuthToken:     "test-token",
			ModelBaseURL:  "http://localhost:8080",
			ModelAPIKey:   "not-needed",
			ModelName:     "auto",
			MemoryBackend: "remote",
		},
	}

	if err := writeOpenClawConfig(configPath, req); err != nil {
		t.Fatalf("writeOpenClawConfig should recover from directory path: %v", err)
	}

	info, err := os.Stat(configPath)
	if err != nil {
		t.Fatalf("failed to stat config path: %v", err)
	}
	if info.IsDir() {
		t.Fatalf("config path should be a file, got directory")
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read config file: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("config file should be valid JSON: %v", err)
	}
}

func TestWriteOpenClawConfig_RemoteMemoryUsesCurrentSchema(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "openclaw.json")

	req := ProvisionRequest{
		Container: ContainerConfig{
			GatewayPort:   18788,
			AuthToken:     "test-token",
			ModelBaseURL:  "http://localhost:8080",
			ModelAPIKey:   "test-api-key",
			ModelName:     "auto",
			MemoryBackend: "remote",
			MemoryBaseURL: "http://127.0.0.1:8080",
			VectorStore:   "legacy-ignored",
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

	memory, ok := cfg["memory"].(map[string]interface{})
	if !ok {
		t.Fatalf("memory block missing or invalid")
	}
	if memory["backend"] != "builtin" {
		t.Fatalf("memory.backend should be builtin for remote embeddings mode, got: %v", memory["backend"])
	}
	if _, legacyRemotePresent := memory["remote"]; legacyRemotePresent {
		t.Fatalf("legacy memory.remote key should not be generated")
	}

	agents, ok := cfg["agents"].(map[string]interface{})
	if !ok {
		t.Fatalf("agents block missing or invalid")
	}
	defaults, ok := agents["defaults"].(map[string]interface{})
	if !ok {
		t.Fatalf("agents.defaults block missing or invalid")
	}
	memorySearch, ok := defaults["memorySearch"].(map[string]interface{})
	if !ok {
		t.Fatalf("agents.defaults.memorySearch should be present for remote mode")
	}
	if memorySearch["provider"] != "openai" {
		t.Fatalf("memorySearch.provider should be openai, got: %v", memorySearch["provider"])
	}

	remote, ok := memorySearch["remote"].(map[string]interface{})
	if !ok {
		t.Fatalf("memorySearch.remote should be present")
	}
	if remote["baseUrl"] != "http://127.0.0.1:8080" {
		t.Fatalf("memorySearch.remote.baseUrl mismatch: %v", remote["baseUrl"])
	}
}

func TestLoadSkills_UsesEnvOverridePath(t *testing.T) {
	tempDir := t.TempDir()
	skillsPath := filepath.Join(tempDir, "openclaw-skills.json")
	if err := os.WriteFile(skillsPath, []byte(`[
  {
    "id": "test-skill",
    "name": "Test Skill",
    "description": "for unit test",
    "emoji": "🧪",
    "category": "test",
    "builtin": true
  }
]`), 0o644); err != nil {
		t.Fatalf("failed to write test skills file: %v", err)
	}

	t.Setenv("OPENCLAW_SKILLS_PATH", skillsPath)
	h := NewOpenClawHandler(filepath.Join(tempDir, "openclaw-data"), false)
	skills, err := h.loadSkills()
	if err != nil {
		t.Fatalf("loadSkills failed: %v", err)
	}
	if len(skills) != 1 {
		t.Fatalf("expected 1 skill, got %d", len(skills))
	}
	if skills[0].ID != "test-skill" {
		t.Fatalf("unexpected skill ID: %s", skills[0].ID)
	}
}

func TestDeriveContainerName(t *testing.T) {
	tests := []struct {
		name         string
		requested    string
		identityName string
		expected     string
	}{
		{
			name:         "requested name wins",
			requested:    "My-Agent_01",
			identityName: "Atlas",
			expected:     "my-agent_01",
		},
		{
			name:         "fallback to identity",
			requested:    "",
			identityName: "Atlas Bot",
			expected:     "atlas-bot",
		},
		{
			name:         "fallback to default when both empty/invalid",
			requested:    "   ",
			identityName: "🦞🦞",
			expected:     "openclaw-demo",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := deriveContainerName(tc.requested, tc.identityName)
			if got != tc.expected {
				t.Fatalf("deriveContainerName(%q, %q) = %q, expected %q", tc.requested, tc.identityName, got, tc.expected)
			}
		})
	}
}
