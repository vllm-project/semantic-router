package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
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
			expected:     "openclaw-vllm-sr",
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

func TestWriteIdentityFiles_VibeIsIncludedInSoulAndIdentity(t *testing.T) {
	tempDir := t.TempDir()
	id := IdentityConfig{
		Name:       "Atlas",
		Role:       "ops agent",
		Vibe:       "calm and precise",
		Principles: "be rigorous",
		UserName:   "Platform Team",
	}

	if err := writeIdentityFiles(tempDir, id); err != nil {
		t.Fatalf("writeIdentityFiles failed: %v", err)
	}

	soulData, err := os.ReadFile(filepath.Join(tempDir, "SOUL.md"))
	if err != nil {
		t.Fatalf("failed to read SOUL.md: %v", err)
	}
	identityData, err := os.ReadFile(filepath.Join(tempDir, "IDENTITY.md"))
	if err != nil {
		t.Fatalf("failed to read IDENTITY.md: %v", err)
	}

	if !strings.Contains(string(soulData), "## Vibe") || !strings.Contains(string(soulData), id.Vibe) {
		t.Fatalf("SOUL.md should include vibe section with value %q", id.Vibe)
	}
	if !strings.Contains(string(identityData), "- **Vibe:** "+id.Vibe) {
		t.Fatalf("IDENTITY.md should include vibe line")
	}
}

func TestAgentsMdContent_IncludesIdentityReadStep(t *testing.T) {
	content := agentsMdContent()
	if !strings.Contains(content, "`IDENTITY.md`") {
		t.Fatalf("AGENTS.md content should instruct reading IDENTITY.md")
	}
}

func TestTeamsHandler_CreateAndList(t *testing.T) {
	tempDir := t.TempDir()
	h := NewOpenClawHandler(tempDir, false)

	createReq := httptest.NewRequest(http.MethodPost, "/api/openclaw/teams", strings.NewReader(`{
		"name":"Research",
		"vibe":"Calm",
		"role":"Routing Team",
		"principal":"Safety first"
	}`))
	createResp := httptest.NewRecorder()
	h.TeamsHandler().ServeHTTP(createResp, createReq)
	if createResp.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", createResp.Code, createResp.Body.String())
	}

	var created TeamEntry
	if err := json.Unmarshal(createResp.Body.Bytes(), &created); err != nil {
		t.Fatalf("failed to parse create response: %v", err)
	}
	if created.ID == "" || created.Name != "Research" {
		t.Fatalf("unexpected team payload: %+v", created)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/api/openclaw/teams", nil)
	listResp := httptest.NewRecorder()
	h.TeamsHandler().ServeHTTP(listResp, listReq)
	if listResp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", listResp.Code)
	}

	var teams []TeamEntry
	if err := json.Unmarshal(listResp.Body.Bytes(), &teams); err != nil {
		t.Fatalf("failed to parse list response: %v", err)
	}
	if len(teams) != 1 {
		t.Fatalf("expected 1 team, got %d", len(teams))
	}
	if teams[0].ID != created.ID {
		t.Fatalf("expected team ID %q, got %q", created.ID, teams[0].ID)
	}
}

func TestTeamByIDHandler_UpdatePropagatesRegistryTeamName(t *testing.T) {
	tempDir := t.TempDir()
	h := NewOpenClawHandler(tempDir, false)

	if err := h.saveTeams([]TeamEntry{{
		ID:        "routing-core",
		Name:      "Routing Core",
		CreatedAt: "2026-01-01T00:00:00Z",
		UpdatedAt: "2026-01-01T00:00:00Z",
	}}); err != nil {
		t.Fatalf("failed to seed teams: %v", err)
	}
	if err := h.saveRegistry([]ContainerEntry{{
		Name:     "agent-1",
		Port:     18788,
		Image:    "ghcr.io/openclaw/openclaw:latest",
		Token:    "token",
		DataDir:  tempDir,
		TeamID:   "routing-core",
		TeamName: "Routing Core",
	}}); err != nil {
		t.Fatalf("failed to seed registry: %v", err)
	}

	updateReq := httptest.NewRequest(http.MethodPut, "/api/openclaw/teams/routing-core", strings.NewReader(`{
		"name":"Routing Core Plus",
		"vibe":"Focused",
		"role":"Routing",
		"principal":"Consistency"
	}`))
	updateResp := httptest.NewRecorder()
	h.TeamByIDHandler().ServeHTTP(updateResp, updateReq)
	if updateResp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", updateResp.Code, updateResp.Body.String())
	}

	entries, err := h.loadRegistry()
	if err != nil {
		t.Fatalf("failed to load registry: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 registry entry, got %d", len(entries))
	}
	if entries[0].TeamName != "Routing Core Plus" {
		t.Fatalf("expected updated team name to propagate to registry, got %q", entries[0].TeamName)
	}
}

func TestTeamByIDHandler_DeleteRejectsAssignedTeam(t *testing.T) {
	tempDir := t.TempDir()
	h := NewOpenClawHandler(tempDir, false)

	if err := h.saveTeams([]TeamEntry{{
		ID:        "alpha",
		Name:      "Alpha",
		CreatedAt: "2026-01-01T00:00:00Z",
		UpdatedAt: "2026-01-01T00:00:00Z",
	}}); err != nil {
		t.Fatalf("failed to seed teams: %v", err)
	}
	if err := h.saveRegistry([]ContainerEntry{{
		Name:     "agent-1",
		Port:     18788,
		Image:    "ghcr.io/openclaw/openclaw:latest",
		Token:    "token",
		DataDir:  tempDir,
		TeamID:   "alpha",
		TeamName: "Alpha",
	}}); err != nil {
		t.Fatalf("failed to seed registry: %v", err)
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/api/openclaw/teams/alpha", nil)
	deleteResp := httptest.NewRecorder()
	h.TeamByIDHandler().ServeHTTP(deleteResp, deleteReq)
	if deleteResp.Code != http.StatusConflict {
		t.Fatalf("expected 409 when team is assigned, got %d", deleteResp.Code)
	}
}

func TestWorkersHandler_List(t *testing.T) {
	tempDir := t.TempDir()
	h := NewOpenClawHandler(tempDir, false)

	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:      "atlas",
			Port:      18788,
			Image:     "ghcr.io/openclaw/openclaw:latest",
			Token:     "token",
			DataDir:   tempDir,
			TeamID:    "research",
			TeamName:  "Research",
			AgentName: "Atlas",
		},
		{
			Name:      "claude",
			Port:      18789,
			Image:     "ghcr.io/openclaw/openclaw:latest",
			Token:     "token",
			DataDir:   tempDir,
			TeamID:    "infra",
			TeamName:  "Infra",
			AgentName: "Claude",
		},
	}); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/openclaw/workers", nil)
	resp := httptest.NewRecorder()
	h.WorkersHandler().ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var workers []ContainerEntry
	if err := json.Unmarshal(resp.Body.Bytes(), &workers); err != nil {
		t.Fatalf("failed to parse workers list: %v", err)
	}
	if len(workers) != 2 {
		t.Fatalf("expected 2 workers, got %d", len(workers))
	}
	if workers[0].Name != "atlas" {
		t.Fatalf("expected sorted worker list by name, got first=%q", workers[0].Name)
	}
}

func TestWorkerByIDHandler_Update(t *testing.T) {
	tempDir := t.TempDir()
	h := NewOpenClawHandler(tempDir, false)

	if err := h.saveTeams([]TeamEntry{
		{ID: "research", Name: "Research", CreatedAt: "2026-01-01T00:00:00Z", UpdatedAt: "2026-01-01T00:00:00Z"},
		{ID: "infra", Name: "Infrastructure", CreatedAt: "2026-01-01T00:00:00Z", UpdatedAt: "2026-01-01T00:00:00Z"},
	}); err != nil {
		t.Fatalf("failed to seed teams: %v", err)
	}
	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:            "atlas",
			Port:            18788,
			Image:           "ghcr.io/openclaw/openclaw:latest",
			Token:           "token",
			DataDir:         tempDir,
			TeamID:          "research",
			TeamName:        "Research",
			AgentName:       "Atlas",
			AgentEmoji:      "🦀",
			AgentRole:       "Researcher",
			AgentVibe:       "Calm",
			AgentPrinciples: "Safety first",
		},
	}); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}

	updateReq := httptest.NewRequest(http.MethodPut, "/api/openclaw/workers/atlas", strings.NewReader(`{
		"teamId":"infra",
		"identity":{
			"name":"Atlas Prime",
			"emoji":"🧠",
			"role":"AI Infra",
			"vibe":"Focused",
			"principles":"Reliability first"
		}
	}`))
	updateResp := httptest.NewRecorder()
	h.WorkerByIDHandler().ServeHTTP(updateResp, updateReq)
	if updateResp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", updateResp.Code, updateResp.Body.String())
	}

	entries, err := h.loadRegistry()
	if err != nil {
		t.Fatalf("failed to load registry: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 worker in registry, got %d", len(entries))
	}
	if entries[0].TeamID != "infra" || entries[0].TeamName != "Infrastructure" {
		t.Fatalf("expected updated team mapping, got id=%q name=%q", entries[0].TeamID, entries[0].TeamName)
	}
	if entries[0].AgentName != "Atlas Prime" || entries[0].AgentRole != "AI Infra" || entries[0].AgentVibe != "Focused" {
		t.Fatalf("identity fields were not updated: %+v", entries[0])
	}
}

func TestWorkerByIDHandler_Delete(t *testing.T) {
	tempDir := t.TempDir()
	h := NewOpenClawHandler(tempDir, false)

	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:    "atlas",
			Port:    18788,
			Image:   "ghcr.io/openclaw/openclaw:latest",
			Token:   "token",
			DataDir: tempDir,
		},
	}); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/api/openclaw/workers/atlas", nil)
	deleteResp := httptest.NewRecorder()
	h.WorkerByIDHandler().ServeHTTP(deleteResp, deleteReq)
	if deleteResp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", deleteResp.Code, deleteResp.Body.String())
	}

	entries, err := h.loadRegistry()
	if err != nil {
		t.Fatalf("failed to load registry after delete: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("expected registry to be empty after delete, got %d entries", len(entries))
	}
}
