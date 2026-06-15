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
