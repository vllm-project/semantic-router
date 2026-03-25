package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
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

	gateway, ok := cfg["gateway"].(map[string]interface{})
	if !ok {
		t.Fatalf("gateway block missing or invalid")
	}
	httpCfg, ok := gateway["http"].(map[string]interface{})
	if !ok {
		t.Fatalf("gateway.http block missing or invalid")
	}
	endpoints, ok := httpCfg["endpoints"].(map[string]interface{})
	if !ok {
		t.Fatalf("gateway.http.endpoints block missing or invalid")
	}
	chatCompletions, ok := endpoints["chatCompletions"].(map[string]interface{})
	if !ok || chatCompletions["enabled"] != true {
		t.Fatalf("chatCompletions endpoint must be enabled by default, got: %#v", endpoints["chatCompletions"])
	}
}

func TestGatewayTokenForContainer_PrefersConfigTokenOverRegistry(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	workerName := "worker-a"

	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:  workerName,
			Token: "registry-token",
		},
	}); err != nil {
		t.Fatalf("failed to seed registry: %v", err)
	}

	configPath := filepath.Join(h.containerDataDir(workerName), "openclaw.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}
	if err := os.WriteFile(configPath, []byte(`{"gateway":{"auth":{"token":"config-token"}}}`), 0o644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	if got := h.gatewayTokenForContainer(workerName); got != "config-token" {
		t.Fatalf("expected config token, got %q", got)
	}
}

func TestGatewayTokenForContainer_FallsBackToRegistryToken(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	workerName := "worker-b"

	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:  workerName,
			Token: "registry-token-only",
		},
	}); err != nil {
		t.Fatalf("failed to seed registry: %v", err)
	}

	if got := h.gatewayTokenForContainer(workerName); got != "registry-token-only" {
		t.Fatalf("expected registry token fallback, got %q", got)
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
	h := newTestOpenClawHandler(t, filepath.Join(tempDir, "openclaw-data"), false)
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

func TestIsOpenClawGatewayPortConflict(t *testing.T) {
	tests := []struct {
		name     string
		logs     string
		port     int
		expected bool
	}{
		{
			name: "openclaw gateway already listening",
			logs: "Gateway failed to start: another gateway instance is already listening on ws://0.0.0.0:18792",
			port: 18792, expected: true,
		},
		{
			name: "generic address already in use",
			logs: "listen tcp 0.0.0.0:18792: bind: address already in use",
			port: 18792, expected: true,
		},
		{
			name: "port already in use",
			logs: "Port 18792 is already in use.",
			port: 18792, expected: true,
		},
		{
			name: "unrelated error",
			logs: "failed to pull image",
			port: 18792, expected: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isOpenClawGatewayPortConflict(tc.logs, tc.port); got != tc.expected {
				t.Fatalf("isOpenClawGatewayPortConflict() = %v, expected %v", got, tc.expected)
			}
		})
	}
}

func TestIsBridgeNetwork(t *testing.T) {
	tests := []struct {
		name        string
		networkMode string
		expected    bool
	}{
		{name: "empty string", networkMode: "", expected: false},
		{name: "host mode", networkMode: "host", expected: false},
		{name: "host mode uppercase", networkMode: "HOST", expected: false},
		{name: "container mode", networkMode: "container:abc", expected: false},
		{name: "container mode uppercase", networkMode: "Container:xyz", expected: false},
		{name: "default bridge", networkMode: "bridge", expected: true},
		{name: "user-defined bridge", networkMode: "vllm-sr-net", expected: true},
		{name: "custom network", networkMode: "my-network", expected: true},
		{name: "with spaces", networkMode: "  vllm-sr-net  ", expected: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isBridgeNetwork(tc.networkMode); got != tc.expected {
				t.Fatalf("isBridgeNetwork(%q) = %v, expected %v", tc.networkMode, got, tc.expected)
			}
		})
	}
}

func TestNextAvailablePortBridgeMode(t *testing.T) {
	// In bridge mode, should always return the fixed port
	h := &OpenClawHandler{}

	bridgeModes := []string{"vllm-sr-net", "bridge", "my-custom-network"}
	for _, nm := range bridgeModes {
		port := h.nextAvailablePort(nm)
		if port != defaultBridgeGatewayPort {
			t.Errorf("nextAvailablePort(%q) = %d, expected %d (fixed bridge port)",
				nm, port, defaultBridgeGatewayPort)
		}
	}
}

func TestOpenClawGatewayListeningReady(t *testing.T) {
	tests := []struct {
		name     string
		logs     string
		expected bool
	}{
		{
			name:     "success listening marker",
			logs:     "[gateway] listening on ws://0.0.0.0:18796",
			expected: true,
		},
		{
			name:     "failure after success",
			logs:     "[gateway] listening on ws://0.0.0.0:18796\nGateway failed to start: another gateway instance is already listening on ws://0.0.0.0:18796",
			expected: false,
		},
		{
			name:     "success after earlier failure",
			logs:     "Gateway failed to start: another gateway instance is already listening on ws://0.0.0.0:18796\n[gateway] listening on ws://0.0.0.0:18797",
			expected: true,
		},
		{
			name:     "no listening marker",
			logs:     "starting openclaw...",
			expected: false,
		},
		{
			name:     "empty logs",
			logs:     "",
			expected: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := openClawGatewayListeningReady(tc.logs); got != tc.expected {
				t.Fatalf("openClawGatewayListeningReady() = %v, expected %v", got, tc.expected)
			}
		})
	}
}

func TestProvisionAsyncRequested(t *testing.T) {
	tests := []struct {
		name     string
		url      string
		header   string
		expected bool
	}{
		{
			name:     "query true",
			url:      "/api/openclaw/workers?async=true",
			expected: true,
		},
		{
			name:     "query one",
			url:      "/api/openclaw/workers?async=1",
			expected: true,
		},
		{
			name:     "header true",
			url:      "/api/openclaw/workers",
			header:   "true",
			expected: true,
		},
		{
			name:     "header on",
			url:      "/api/openclaw/workers",
			header:   "on",
			expected: true,
		},
		{
			name:     "disabled",
			url:      "/api/openclaw/workers?async=false",
			expected: false,
		},
		{
			name:     "missing",
			url:      "/api/openclaw/workers",
			expected: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tc.url, nil)
			if tc.header != "" {
				req.Header.Set("X-OpenClaw-Async", tc.header)
			}
			if got := provisionAsyncRequested(req); got != tc.expected {
				t.Fatalf("provisionAsyncRequested() = %v, expected %v", got, tc.expected)
			}
		})
	}
}

func TestResolveOpenClawModelBaseURL_UsesRouterListeners(t *testing.T) {
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

	h := newTestOpenClawHandler(t, tempDir, false)
	h.SetRouterConfigPath(configPath)
	t.Setenv("OPENCLAW_MODEL_BASE_URL", "")

	if got := h.resolveOpenClawModelBaseURL(); got != "http://127.0.0.1:18889/v1" {
		t.Fatalf("expected listener-derived model base URL, got %q", got)
	}
}

func TestResolveOpenClawModelBaseURL_EnvOverrideWins(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	configYAML := `
api_server:
  listeners:
    - address: ::1
      port: 18890
`
	if err := os.WriteFile(configPath, []byte(configYAML), 0o644); err != nil {
		t.Fatalf("failed to write config file: %v", err)
	}

	h := newTestOpenClawHandler(t, tempDir, false)
	h.SetRouterConfigPath(configPath)
	t.Setenv("OPENCLAW_MODEL_BASE_URL", "http://localhost:19999/v1")

	if got := h.resolveOpenClawModelBaseURL(); got != "http://localhost:19999/v1" {
		t.Fatalf("expected env override model base URL, got %q", got)
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
	h := newTestOpenClawHandler(t, tempDir, false)

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
	h := newTestOpenClawHandler(t, tempDir, false)

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
	h := newTestOpenClawHandler(t, tempDir, false)

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
	h := newTestOpenClawHandler(t, tempDir, false)

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
	h := newTestOpenClawHandler(t, tempDir, false)

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
	h := newTestOpenClawHandler(t, tempDir, false)

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

func TestWorkerByIDHandler_SetLeaderRoleUpdatesTeamAndDemotesOthers(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	if err := h.saveTeams([]TeamEntry{
		{
			ID:        "core",
			Name:      "Core Team",
			LeaderID:  "worker-a",
			CreatedAt: "2026-01-01T00:00:00Z",
			UpdatedAt: "2026-01-01T00:00:00Z",
		},
	}); err != nil {
		t.Fatalf("failed to seed teams: %v", err)
	}
	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:     "worker-a",
			Port:     18788,
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "token",
			DataDir:  tempDir,
			TeamID:   "core",
			TeamName: "Core Team",
			RoleKind: "leader",
		},
		{
			Name:     "worker-b",
			Port:     18789,
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "token",
			DataDir:  tempDir,
			TeamID:   "core",
			TeamName: "Core Team",
			RoleKind: "worker",
		},
	}); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}

	updateReq := httptest.NewRequest(http.MethodPut, "/api/openclaw/workers/worker-b", strings.NewReader(`{
		"roleKind":"leader"
	}`))
	updateResp := httptest.NewRecorder()
	h.WorkerByIDHandler().ServeHTTP(updateResp, updateReq)
	if updateResp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", updateResp.Code, updateResp.Body.String())
	}

	teams, err := h.loadTeams()
	if err != nil {
		t.Fatalf("failed to load teams: %v", err)
	}
	if len(teams) != 1 || teams[0].LeaderID != "worker-b" {
		t.Fatalf("expected team leader to become worker-b, got %+v", teams)
	}

	entries, err := h.loadRegistry()
	if err != nil {
		t.Fatalf("failed to load workers: %v", err)
	}
	roleByName := map[string]string{}
	leaderRole := ""
	leaderVibe := ""
	leaderPrinciples := ""
	for _, entry := range entries {
		roleByName[entry.Name] = normalizeRoleKind(entry.RoleKind)
		if entry.Name == "worker-b" {
			leaderRole = strings.TrimSpace(entry.AgentRole)
			leaderVibe = strings.TrimSpace(entry.AgentVibe)
			leaderPrinciples = strings.TrimSpace(entry.AgentPrinciples)
		}
	}
	if roleByName["worker-b"] != "leader" {
		t.Fatalf("worker-b should be leader, got %q", roleByName["worker-b"])
	}
	if roleByName["worker-a"] != "worker" {
		t.Fatalf("worker-a should be demoted to worker, got %q", roleByName["worker-a"])
	}
	if leaderRole == "" || leaderVibe == "" || leaderPrinciples == "" {
		t.Fatalf("leader metadata defaults should be populated, got role=%q vibe=%q principles=%q", leaderRole, leaderVibe, leaderPrinciples)
	}
}

func TestRoomsHandler_CreateAndMessageFlowWithoutAutomation(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	if err := h.saveTeams([]TeamEntry{{
		ID:        "team-a",
		Name:      "Team A",
		CreatedAt: "2026-01-01T00:00:00Z",
		UpdatedAt: "2026-01-01T00:00:00Z",
	}}); err != nil {
		t.Fatalf("failed to seed team: %v", err)
	}

	createRoomReq := httptest.NewRequest(http.MethodPost, "/api/openclaw/rooms", strings.NewReader(`{
		"teamId":"team-a",
		"name":"Planning"
	}`))
	createRoomResp := httptest.NewRecorder()
	h.RoomsHandler().ServeHTTP(createRoomResp, createRoomReq)
	if createRoomResp.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", createRoomResp.Code, createRoomResp.Body.String())
	}
	var room ClawRoomEntry
	if err := json.Unmarshal(createRoomResp.Body.Bytes(), &room); err != nil {
		t.Fatalf("failed to parse room create response: %v", err)
	}
	if room.ID == "" {
		t.Fatalf("room id should not be empty")
	}

	listReq := httptest.NewRequest(http.MethodGet, "/api/openclaw/rooms?teamId=team-a", nil)
	listResp := httptest.NewRecorder()
	h.RoomsHandler().ServeHTTP(listResp, listReq)
	if listResp.Code != http.StatusOK {
		t.Fatalf("expected 200 for room list, got %d", listResp.Code)
	}

	postReq := httptest.NewRequest(http.MethodPost, fmt.Sprintf("/api/openclaw/rooms/%s/messages", room.ID), strings.NewReader(`{
		"senderType":"system",
		"senderName":"test",
		"content":"hello @leader"
	}`))
	postResp := httptest.NewRecorder()
	h.RoomByIDHandler().ServeHTTP(postResp, postReq)
	if postResp.Code != http.StatusCreated {
		t.Fatalf("expected 201 for message post, got %d: %s", postResp.Code, postResp.Body.String())
	}

	msgListReq := httptest.NewRequest(http.MethodGet, fmt.Sprintf("/api/openclaw/rooms/%s/messages?limit=10", room.ID), nil)
	msgListResp := httptest.NewRecorder()
	h.RoomByIDHandler().ServeHTTP(msgListResp, msgListReq)
	if msgListResp.Code != http.StatusOK {
		t.Fatalf("expected 200 for message list, got %d", msgListResp.Code)
	}
	var messages []ClawRoomMessage
	if err := json.Unmarshal(msgListResp.Body.Bytes(), &messages); err != nil {
		t.Fatalf("failed to parse message list: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if len(messages[0].Mentions) != 1 || messages[0].Mentions[0] != "leader" {
		t.Fatalf("mention parsing mismatch: %+v", messages[0].Mentions)
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, fmt.Sprintf("/api/openclaw/rooms/%s", room.ID), nil)
	deleteResp := httptest.NewRecorder()
	h.RoomByIDHandler().ServeHTTP(deleteResp, deleteReq)
	if deleteResp.Code != http.StatusOK {
		t.Fatalf("expected 200 for room delete, got %d: %s", deleteResp.Code, deleteResp.Body.String())
	}

	getRoomReq := httptest.NewRequest(http.MethodGet, fmt.Sprintf("/api/openclaw/rooms/%s", room.ID), nil)
	getRoomResp := httptest.NewRecorder()
	h.RoomByIDHandler().ServeHTTP(getRoomResp, getRoomReq)
	if getRoomResp.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for deleted room, got %d", getRoomResp.Code)
	}
}

func TestOpenClawReadonlyBlocksManagementMutations(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), true)

	testCases := []struct {
		name    string
		method  string
		path    string
		body    string
		handler http.HandlerFunc
	}{
		{name: "team create", method: http.MethodPost, path: "/api/openclaw/teams", body: `{}`, handler: h.TeamsHandler()},
		{name: "team update", method: http.MethodPut, path: "/api/openclaw/teams/team-a", body: `{}`, handler: h.TeamByIDHandler()},
		{name: "team delete", method: http.MethodDelete, path: "/api/openclaw/teams/team-a", handler: h.TeamByIDHandler()},
		{name: "worker create", method: http.MethodPost, path: "/api/openclaw/workers", body: `{}`, handler: h.WorkersHandler()},
		{name: "worker update", method: http.MethodPut, path: "/api/openclaw/workers/worker-a", body: `{}`, handler: h.WorkerByIDHandler()},
		{name: "worker delete", method: http.MethodDelete, path: "/api/openclaw/workers/worker-a", handler: h.WorkerByIDHandler()},
		{name: "room create", method: http.MethodPost, path: "/api/openclaw/rooms", body: `{}`, handler: h.RoomsHandler()},
		{name: "room delete", method: http.MethodDelete, path: "/api/openclaw/rooms/room-a", handler: h.RoomByIDHandler()},
		{name: "start", method: http.MethodPost, path: "/api/openclaw/start", body: `{}`, handler: h.StartHandler()},
		{name: "stop", method: http.MethodPost, path: "/api/openclaw/stop", body: `{}`, handler: h.StopHandler()},
		{name: "remove", method: http.MethodDelete, path: "/api/openclaw/containers/worker-a", handler: h.DeleteHandler()},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var body *strings.Reader
			if tc.body != "" {
				body = strings.NewReader(tc.body)
			} else {
				body = strings.NewReader("")
			}
			req := httptest.NewRequest(tc.method, tc.path, body)
			resp := httptest.NewRecorder()
			tc.handler.ServeHTTP(resp, req)
			if resp.Code != http.StatusForbidden {
				t.Fatalf("expected 403, got %d: %s", resp.Code, resp.Body.String())
			}
			if !strings.Contains(resp.Body.String(), "Read-only mode enabled") {
				t.Fatalf("expected readonly error, got %q", resp.Body.String())
			}
		})
	}
}

func TestRoomsHandler_CreateRoomAutoSuffixAvoidsConflict(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)
	if err := h.saveTeams([]TeamEntry{{
		ID:        "llm-router-lab",
		Name:      "LLM Router Lab",
		CreatedAt: "2026-01-01T00:00:00Z",
		UpdatedAt: "2026-01-01T00:00:00Z",
	}}); err != nil {
		t.Fatalf("failed to seed team: %v", err)
	}

	createWithName := func(name string) ClawRoomEntry {
		req := httptest.NewRequest(http.MethodPost, "/api/openclaw/rooms", strings.NewReader(fmt.Sprintf(`{
			"teamId":"llm-router-lab",
			"name":"%s"
		}`, name)))
		resp := httptest.NewRecorder()
		h.RoomsHandler().ServeHTTP(resp, req)
		if resp.Code != http.StatusCreated {
			t.Fatalf("expected 201, got %d: %s", resp.Code, resp.Body.String())
		}
		var created ClawRoomEntry
		if err := json.Unmarshal(resp.Body.Bytes(), &created); err != nil {
			t.Fatalf("failed to parse room create response: %v", err)
		}
		return created
	}

	first := createWithName("llm-router-lab-room")
	second := createWithName("llm-router-lab-room")
	if first.ID == second.ID {
		t.Fatalf("room IDs should be unique, got duplicated id %q", first.ID)
	}

	baseID := sanitizeRoomID("llm-router-lab-room")
	for _, room := range []ClawRoomEntry{first, second} {
		if !strings.HasPrefix(room.ID, baseID+"-") {
			t.Fatalf("room ID %q should keep fixed base prefix %q", room.ID, baseID+"-")
		}
		suffix := strings.TrimPrefix(room.ID, baseID+"-")
		if len(suffix) < 4 {
			t.Fatalf("room ID %q should include a short dynamic suffix", room.ID)
		}
	}

	duplicateIDReq := httptest.NewRequest(http.MethodPost, "/api/openclaw/rooms", strings.NewReader(fmt.Sprintf(`{
		"teamId":"llm-router-lab",
		"name":"manual",
		"id":"%s"
	}`, first.ID)))
	duplicateIDResp := httptest.NewRecorder()
	h.RoomsHandler().ServeHTTP(duplicateIDResp, duplicateIDReq)
	if duplicateIDResp.Code != http.StatusConflict {
		t.Fatalf("expected explicit duplicate id to return 409, got %d: %s", duplicateIDResp.Code, duplicateIDResp.Body.String())
	}
}

func TestProcessRoomUserMessage_LeaderDelegatesToWorker(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	leaderSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected leader path: %s", r.URL.Path)
		}
		if got := r.Header.Get("X-OpenClaw-Agent-Id"); got != openClawPrimaryAgentID {
			t.Fatalf("expected X-OpenClaw-Agent-Id=%s, got %q", openClawPrimaryAgentID, got)
		}
		var payload openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode leader payload: %v", err)
		}
		if payload.Model != openClawPrimaryAgentModel {
			t.Fatalf("unexpected leader model: %s", payload.Model)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"@worker-a please prepare the implementation checklist."}}]}`))
	}))
	defer leaderSrv.Close()

	workerSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected worker path: %s", r.URL.Path)
		}
		var payload openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode worker payload: %v", err)
		}
		if payload.Model != openClawPrimaryAgentModel {
			t.Fatalf("unexpected worker model: %s", payload.Model)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"Checklist prepared and ready for review."}}]}`))
	}))
	defer workerSrv.Close()

	leaderPort := mustServerPort(t, leaderSrv.URL)
	workerPort := mustServerPort(t, workerSrv.URL)

	team := TeamEntry{
		ID:        "team-alpha",
		Name:      "Alpha",
		LeaderID:  "leader-1",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := h.saveTeams([]TeamEntry{team}); err != nil {
		t.Fatalf("failed to seed team: %v", err)
	}
	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:     "leader-1",
			Port:     leaderPort,
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "leader-token",
			DataDir:  tempDir,
			TeamID:   team.ID,
			TeamName: team.Name,
			RoleKind: "leader",
		},
		{
			Name:     "worker-a",
			Port:     workerPort,
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "worker-token",
			DataDir:  tempDir,
			TeamID:   team.ID,
			TeamName: team.Name,
			RoleKind: "worker",
		},
	}); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}

	h.mu.Lock()
	room, err := h.ensureDefaultRoomLocked(team)
	h.mu.Unlock()
	if err != nil {
		t.Fatalf("failed to ensure room: %v", err)
	}

	userMessage := newRoomMessage(room, "user", "user-1", "You", "Please @leader break this down and delegate.", nil)
	err = h.appendRoomMessage(room.ID, userMessage)
	if err != nil {
		t.Fatalf("failed to append user message: %v", err)
	}

	h.processRoomUserMessage(room.ID, userMessage.ID)

	messages, err := h.loadRoomMessages(room.ID)
	if err != nil {
		t.Fatalf("failed to load room messages: %v", err)
	}
	if len(messages) < 3 {
		t.Fatalf("expected at least 3 room messages (user + leader + worker), got %d", len(messages))
	}

	leaderFound := false
	delegatedFound := false
	for _, msg := range messages {
		if msg.SenderID == "leader-1" && strings.Contains(msg.Content, "@worker-a") {
			leaderFound = true
		}
		if msg.SenderID == "worker-a" && strings.Contains(strings.ToLower(msg.Content), "checklist prepared") {
			delegatedFound = true
		}
	}
	if !leaderFound {
		t.Fatalf("expected leader response mentioning worker-a, got: %+v", messages)
	}
	if !delegatedFound {
		t.Fatalf("expected delegated worker response, got: %+v", messages)
	}
}

func TestProcessRoomUserMessage_SimultaneousMentionsContinueChain(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var (
		callsMu     sync.Mutex
		leaderCalls int
		workerCalls int
	)

	leaderSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected leader path: %s", r.URL.Path)
		}
		var payload openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode leader payload: %v", err)
		}
		callsMu.Lock()
		leaderCalls++
		call := leaderCalls
		callsMu.Unlock()

		content := "Leader reviewed worker output."
		if call == 1 {
			content = "@worker-a please draft the implementation plan."
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": content,
					},
				},
			},
		})
	}))
	defer leaderSrv.Close()

	workerSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected worker path: %s", r.URL.Path)
		}
		var payload openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode worker payload: %v", err)
		}
		callsMu.Lock()
		workerCalls++
		call := workerCalls
		callsMu.Unlock()

		content := "Worker final updates done."
		if call == 1 {
			content = "@leader initial draft is complete."
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": content,
					},
				},
			},
		})
	}))
	defer workerSrv.Close()

	team := TeamEntry{
		ID:        "team-sync",
		Name:      "Sync Team",
		LeaderID:  "leader-1",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := h.saveTeams([]TeamEntry{team}); err != nil {
		t.Fatalf("failed to seed team: %v", err)
	}
	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:     "leader-1",
			Port:     mustServerPort(t, leaderSrv.URL),
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "leader-token",
			DataDir:  tempDir,
			TeamID:   team.ID,
			RoleKind: "leader",
		},
		{
			Name:     "worker-a",
			Port:     mustServerPort(t, workerSrv.URL),
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "worker-token",
			DataDir:  tempDir,
			TeamID:   team.ID,
			RoleKind: "worker",
		},
	}); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}

	h.mu.Lock()
	room, err := h.ensureDefaultRoomLocked(team)
	h.mu.Unlock()
	if err != nil {
		t.Fatalf("failed to ensure room: %v", err)
	}

	userMessage := newRoomMessage(room, "user", "user-1", "You", "Please @leader and @worker-a collaborate.", nil)
	err = h.appendRoomMessage(room.ID, userMessage)
	if err != nil {
		t.Fatalf("failed to append user message: %v", err)
	}

	h.processRoomUserMessage(room.ID, userMessage.ID)

	callsMu.Lock()
	gotLeaderCalls := leaderCalls
	gotWorkerCalls := workerCalls
	callsMu.Unlock()
	if gotLeaderCalls != 1 {
		t.Fatalf("expected leader to be called exactly once (worker @leader should be ignored), got %d", gotLeaderCalls)
	}
	if gotWorkerCalls < 2 {
		t.Fatalf("expected worker to be called at least twice, got %d", gotWorkerCalls)
	}

	messages, err := h.loadRoomMessages(room.ID)
	if err != nil {
		t.Fatalf("failed to load room messages: %v", err)
	}
	if len(messages) < 4 {
		t.Fatalf("expected at least 4 room messages, got %d", len(messages))
	}

	leaderMentionedWorker := false
	workerMentionedLeader := false
	for _, msg := range messages {
		if msg.SenderID == "leader-1" && strings.Contains(msg.Content, "@worker-a") {
			leaderMentionedWorker = true
		}
		if msg.SenderID == "worker-a" && strings.Contains(strings.ToLower(msg.Content), "@leader") {
			workerMentionedLeader = true
		}
	}
	if !leaderMentionedWorker {
		t.Fatalf("expected leader message that mentions worker-a, got: %+v", messages)
	}
	if !workerMentionedLeader {
		t.Fatalf("expected worker message that mentions leader, got: %+v", messages)
	}
}

func TestProcessRoomUserMessage_MentionAllTargetsEntireTeam(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var (
		callsMu      sync.Mutex
		leaderCalls  int
		workerACalls int
		workerBCalls int
	)

	leaderSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected leader path: %s", r.URL.Path)
		}
		callsMu.Lock()
		leaderCalls++
		callsMu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "Leader acknowledged.",
					},
				},
			},
		})
	}))
	defer leaderSrv.Close()

	workerASrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected worker-a path: %s", r.URL.Path)
		}
		callsMu.Lock()
		workerACalls++
		callsMu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "Worker A acknowledged.",
					},
				},
			},
		})
	}))
	defer workerASrv.Close()

	workerBSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected worker-b path: %s", r.URL.Path)
		}
		callsMu.Lock()
		workerBCalls++
		callsMu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "Worker B acknowledged.",
					},
				},
			},
		})
	}))
	defer workerBSrv.Close()

	team := TeamEntry{
		ID:        "team-all",
		Name:      "All Team",
		LeaderID:  "leader-1",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := h.saveTeams([]TeamEntry{team}); err != nil {
		t.Fatalf("failed to seed team: %v", err)
	}
	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:     "leader-1",
			Port:     mustServerPort(t, leaderSrv.URL),
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "leader-token",
			DataDir:  tempDir,
			TeamID:   team.ID,
			RoleKind: "leader",
		},
		{
			Name:     "worker-a",
			Port:     mustServerPort(t, workerASrv.URL),
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "worker-a-token",
			DataDir:  tempDir,
			TeamID:   team.ID,
			RoleKind: "worker",
		},
		{
			Name:     "worker-b",
			Port:     mustServerPort(t, workerBSrv.URL),
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "worker-b-token",
			DataDir:  tempDir,
			TeamID:   team.ID,
			RoleKind: "worker",
		},
	}); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}

	h.mu.Lock()
	room, err := h.ensureDefaultRoomLocked(team)
	h.mu.Unlock()
	if err != nil {
		t.Fatalf("failed to ensure room: %v", err)
	}

	userMessage := newRoomMessage(
		room,
		"user",
		"user-1",
		"You",
		"@all please share your current status.",
		nil,
	)
	if err := h.appendRoomMessage(room.ID, userMessage); err != nil {
		t.Fatalf("failed to append user message: %v", err)
	}

	h.processRoomUserMessage(room.ID, userMessage.ID)

	callsMu.Lock()
	gotLeader := leaderCalls
	gotWorkerA := workerACalls
	gotWorkerB := workerBCalls
	callsMu.Unlock()
	if gotLeader != 1 {
		t.Fatalf("expected leader to be called once for @all, got %d", gotLeader)
	}
	if gotWorkerA != 1 {
		t.Fatalf("expected worker-a to be called once for @all, got %d", gotWorkerA)
	}
	if gotWorkerB != 1 {
		t.Fatalf("expected worker-b to be called once for @all, got %d", gotWorkerB)
	}
}

func TestProcessRoomUserMessage_DuplicateTriggerDoesNotReprocess(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var (
		callsMu    sync.Mutex
		workerCall int
	)

	workerSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected worker path: %s", r.URL.Path)
		}
		callsMu.Lock()
		workerCall++
		callsMu.Unlock()

		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "Done.",
					},
				},
			},
		})
	}))
	defer workerSrv.Close()

	team := TeamEntry{
		ID:        "team-dedup",
		Name:      "Dedup Team",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := h.saveTeams([]TeamEntry{team}); err != nil {
		t.Fatalf("failed to seed team: %v", err)
	}
	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:     "worker-a",
			Port:     mustServerPort(t, workerSrv.URL),
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "token-worker-a",
			DataDir:  tempDir,
			TeamID:   team.ID,
			RoleKind: "worker",
		},
	}); err != nil {
		t.Fatalf("failed to seed worker: %v", err)
	}

	h.mu.Lock()
	room, err := h.ensureDefaultRoomLocked(team)
	h.mu.Unlock()
	if err != nil {
		t.Fatalf("failed to ensure room: %v", err)
	}

	trigger := newRoomMessage(room, "user", "user-1", "You", "@worker-a run once", nil)
	err = h.appendRoomMessage(room.ID, trigger)
	if err != nil {
		t.Fatalf("failed to append trigger message: %v", err)
	}

	h.processRoomUserMessage(room.ID, trigger.ID)
	h.processRoomUserMessage(room.ID, trigger.ID)

	callsMu.Lock()
	gotCalls := workerCall
	callsMu.Unlock()
	if gotCalls != 1 {
		t.Fatalf("expected worker to be called once for duplicate trigger id, got %d", gotCalls)
	}

	messages, err := h.loadRoomMessages(room.ID)
	if err != nil {
		t.Fatalf("failed to load room messages: %v", err)
	}
	var triggerStored *ClawRoomMessage
	for i := range messages {
		if messages[i].ID == trigger.ID {
			triggerStored = &messages[i]
			break
		}
	}
	if triggerStored == nil {
		t.Fatalf("trigger message not found in room history")
	}
	if triggerStored.Metadata == nil || strings.TrimSpace(triggerStored.Metadata[roomAutomationProcessedAtKey]) == "" {
		t.Fatalf("trigger message should be marked as processed, got metadata: %+v", triggerStored.Metadata)
	}
}

func TestProcessRoomUserMessage_MultiMentionsDispatchInParallel(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	var (
		startMu sync.Mutex
		startA  time.Time
		startB  time.Time
	)

	workerASrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected worker-a path: %s", r.URL.Path)
		}
		startMu.Lock()
		startA = time.Now()
		startMu.Unlock()

		time.Sleep(300 * time.Millisecond)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "worker-a done",
					},
				},
			},
		})
	}))
	defer workerASrv.Close()

	workerBSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected worker-b path: %s", r.URL.Path)
		}
		startMu.Lock()
		startB = time.Now()
		startMu.Unlock()

		time.Sleep(300 * time.Millisecond)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "worker-b done",
					},
				},
			},
		})
	}))
	defer workerBSrv.Close()

	team := TeamEntry{
		ID:        "team-parallel",
		Name:      "Parallel Team",
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	if err := h.saveTeams([]TeamEntry{team}); err != nil {
		t.Fatalf("failed to seed team: %v", err)
	}
	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:     "worker-a",
			Port:     mustServerPort(t, workerASrv.URL),
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "token-a",
			DataDir:  tempDir,
			TeamID:   team.ID,
			RoleKind: "worker",
		},
		{
			Name:     "worker-b",
			Port:     mustServerPort(t, workerBSrv.URL),
			Image:    "ghcr.io/openclaw/openclaw:latest",
			Token:    "token-b",
			DataDir:  tempDir,
			TeamID:   team.ID,
			RoleKind: "worker",
		},
	}); err != nil {
		t.Fatalf("failed to seed workers: %v", err)
	}

	h.mu.Lock()
	room, err := h.ensureDefaultRoomLocked(team)
	h.mu.Unlock()
	if err != nil {
		t.Fatalf("failed to ensure room: %v", err)
	}

	userMessage := newRoomMessage(room, "user", "user-1", "You", "@worker-a @worker-b please run in parallel", nil)
	if err := h.appendRoomMessage(room.ID, userMessage); err != nil {
		t.Fatalf("failed to append user message: %v", err)
	}

	h.processRoomUserMessage(room.ID, userMessage.ID)

	startMu.Lock()
	defer startMu.Unlock()
	if startA.IsZero() || startB.IsZero() {
		t.Fatalf("expected both worker requests to be started, got startA=%v startB=%v", startA, startB)
	}
	diff := startA.Sub(startB)
	if diff < 0 {
		diff = -diff
	}
	if diff > 220*time.Millisecond {
		t.Fatalf("expected worker requests to start nearly together for parallel dispatch, diff=%s", diff)
	}
}

func TestQueryWorkerChat_FallsBackToAlternativeEndpoint(t *testing.T) {
	tempDir := t.TempDir()
	h := newTestOpenClawHandler(t, tempDir, false)

	primaryAttempts := 0
	fallbackAttempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions":
			primaryAttempts++
			http.NotFound(w, r)
		case "/api/openai/v1/chat/completions":
			fallbackAttempts++
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"fallback endpoint reply"}}]}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	worker := ContainerEntry{
		Name:     "mira",
		Port:     mustServerPort(t, srv.URL),
		Image:    "ghcr.io/openclaw/openclaw:latest",
		Token:    "test-token",
		DataDir:  tempDir,
		RoleKind: "worker",
	}
	if err := h.saveRegistry([]ContainerEntry{worker}); err != nil {
		t.Fatalf("failed to seed registry: %v", err)
	}

	content, err := h.queryWorkerChat(worker, "system", "user")
	if err != nil {
		t.Fatalf("queryWorkerChat should succeed via fallback endpoint: %v", err)
	}
	if !strings.Contains(content, "fallback endpoint reply") {
		t.Fatalf("unexpected content: %q", content)
	}
	if primaryAttempts == 0 {
		t.Fatalf("expected primary endpoint to be attempted at least once")
	}
	if fallbackAttempts == 0 {
		t.Fatalf("expected fallback endpoint to be attempted at least once")
	}
}

func mustServerPort(t *testing.T, rawURL string) int {
	t.Helper()
	parsed, err := url.Parse(rawURL)
	if err != nil {
		t.Fatalf("failed to parse test server URL: %v", err)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil {
		t.Fatalf("failed to parse test server port: %v", err)
	}
	return port
}
