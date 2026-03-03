package handlers

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// --- Registry ---

type ContainerEntry struct {
	Name      string `json:"name"`
	Port      int    `json:"port"`
	Image     string `json:"image"`
	Token     string `json:"token"`
	DataDir   string `json:"dataDir"`
	CreatedAt string `json:"createdAt"`
}

type OpenClawHandler struct {
	dataDir  string
	readOnly bool
	mu       sync.RWMutex
}

func NewOpenClawHandler(dataDir string, readOnly bool) *OpenClawHandler {
	return &OpenClawHandler{dataDir: dataDir, readOnly: readOnly}
}

func (h *OpenClawHandler) registryPath() string {
	return filepath.Join(h.dataDir, "containers.json")
}

func (h *OpenClawHandler) loadRegistry() ([]ContainerEntry, error) {
	data, err := os.ReadFile(h.registryPath())
	if err != nil {
		if os.IsNotExist(err) {
			return []ContainerEntry{}, nil
		}
		return nil, err
	}
	var entries []ContainerEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}
	return entries, nil
}

func (h *OpenClawHandler) saveRegistry(entries []ContainerEntry) error {
	data, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(h.registryPath()), 0o755); err != nil {
		return err
	}
	return os.WriteFile(h.registryPath(), data, 0o644)
}

func (h *OpenClawHandler) findEntry(name string) *ContainerEntry {
	entries, err := h.loadRegistry()
	if err != nil {
		return nil
	}
	for i := range entries {
		if entries[i].Name == name {
			return &entries[i]
		}
	}
	return nil
}

func (h *OpenClawHandler) nextAvailablePort() int {
	entries, _ := h.loadRegistry()
	used := map[int]bool{}
	for _, e := range entries {
		used[e.Port] = true
	}
	for port := 18788; ; port++ {
		if !used[port] {
			return port
		}
	}
}

// containerDataDir returns the per-container data directory.
func (h *OpenClawHandler) containerDataDir(name string) string {
	return filepath.Join(h.dataDir, "containers", name)
}

// --- Types ---

type SkillTemplate struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Emoji       string   `json:"emoji"`
	Category    string   `json:"category"`
	Builtin     bool     `json:"builtin"`
	Requires    []string `json:"requires,omitempty"`
	OS          []string `json:"os,omitempty"`
}

type IdentityConfig struct {
	Name       string `json:"name"`
	Emoji      string `json:"emoji"`
	Role       string `json:"role"`
	Vibe       string `json:"vibe"`
	Principles string `json:"principles"`
	Boundaries string `json:"boundaries"`
	UserName   string `json:"userName"`
	UserNotes  string `json:"userNotes"`
}

type ContainerConfig struct {
	ContainerName  string `json:"containerName"`
	GatewayPort    int    `json:"gatewayPort"`
	AuthToken      string `json:"authToken"`
	ModelBaseURL   string `json:"modelBaseUrl"`
	ModelAPIKey    string `json:"modelApiKey"`
	ModelName      string `json:"modelName"`
	MemoryBackend  string `json:"memoryBackend"`
	MemoryBaseURL  string `json:"memoryBaseUrl"`
	VectorStore    string `json:"vectorStore"`
	BrowserEnabled bool   `json:"browserEnabled"`
	BaseImage      string `json:"baseImage"`
	NetworkMode    string `json:"networkMode"`
}

type ProvisionRequest struct {
	Identity  IdentityConfig  `json:"identity"`
	Skills    []string        `json:"skills"`
	Container ContainerConfig `json:"container"`
}

type ProvisionResponse struct {
	Success      bool   `json:"success"`
	Message      string `json:"message"`
	WorkspaceDir string `json:"workspaceDir,omitempty"`
	ConfigPath   string `json:"configPath,omitempty"`
	ContainerID  string `json:"containerId,omitempty"`
	DockerCmd    string `json:"dockerCmd,omitempty"`
	ComposeYAML  string `json:"composeYaml,omitempty"`
}

type OpenClawStatus struct {
	Running       bool   `json:"running"`
	ContainerName string `json:"containerName,omitempty"`
	GatewayURL    string `json:"gatewayUrl,omitempty"`
	Port          int    `json:"port,omitempty"`
	Healthy       bool   `json:"healthy"`
	Error         string `json:"error,omitempty"`
}

// --- Token ---

func (h *OpenClawHandler) gatewayTokenForContainer(name string) string {
	entry := h.findEntry(name)
	if entry != nil && entry.Token != "" {
		return entry.Token
	}
	configPath := filepath.Join(h.containerDataDir(name), "openclaw.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}
	var cfg struct {
		Gateway struct {
			Auth struct {
				Token string `json:"token"`
			} `json:"auth"`
		} `json:"gateway"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return ""
	}
	return cfg.Gateway.Auth.Token
}

func (h *OpenClawHandler) TokenHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		name := r.URL.Query().Get("name")
		if name == "" {
			http.Error(w, `{"error":"name parameter required"}`, http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{
			"token": h.gatewayTokenForContainer(name),
		}); err != nil {
			log.Printf("openclaw: token encode error: %v", err)
		}
	}
}

// --- Status ---

func (h *OpenClawHandler) checkContainerHealth(entry ContainerEntry) OpenClawStatus {
	status := OpenClawStatus{
		ContainerName: entry.Name,
		GatewayURL:    fmt.Sprintf("http://localhost:%d", entry.Port),
		Port:          entry.Port,
	}

	out, err := exec.Command("docker", "inspect", "-f", "{{.State.Running}}", entry.Name).Output() // #nosec G204
	if err != nil {
		status.Running = false
		status.Error = "Container not found"
		return status
	}
	status.Running = strings.TrimSpace(string(out)) == "true"
	if !status.Running {
		status.Error = "Container stopped"
		return status
	}

	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/health", entry.Port))
	if err != nil {
		status.Error = "Gateway not reachable"
		return status
	}
	resp.Body.Close()
	gatewayReachable := resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusNoContent

	// Compare positions so a successful restart after a previous failure is correctly detected.
	if logOut, err := exec.Command("docker", "logs", "--tail", "80", entry.Name).CombinedOutput(); err == nil { // #nosec G204
		logs := string(logOut)
		lastSuccess := strings.LastIndex(logs, "[gateway] listening on ws://")
		lastFail := max(
			strings.LastIndex(logs, "failed to start:"),
			strings.LastIndex(logs, "permission denied, mkdir '/state/"),
		)
		if gatewayReachable && lastSuccess >= 0 && lastSuccess > lastFail {
			status.Healthy = true
		} else if lastFail >= 0 && lastFail > lastSuccess {
			status.Healthy = false
			status.Error = "Subsystem initialization failure"
		} else if gatewayReachable {
			status.Healthy = true
		}
	} else if gatewayReachable {
		status.Healthy = true
	}
	if status.Healthy {
		status.Error = ""
	}
	return status
}

func (h *OpenClawHandler) StatusHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		h.mu.RLock()
		entries, err := h.loadRegistry()
		h.mu.RUnlock()
		if err != nil {
			writeJSONError(w, fmt.Sprintf("Failed to load registry: %v", err), http.StatusInternalServerError)
			return
		}

		name := r.URL.Query().Get("name")
		if name != "" {
			for _, e := range entries {
				if e.Name == name {
					w.Header().Set("Content-Type", "application/json")
					if err := json.NewEncoder(w).Encode(h.checkContainerHealth(e)); err != nil {
						log.Printf("openclaw: status encode error: %v", err)
					}
					return
				}
			}
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(OpenClawStatus{Error: "Container not in registry"}); err != nil {
				log.Printf("openclaw: status encode error: %v", err)
			}
			return
		}

		statuses := make([]OpenClawStatus, 0, len(entries))
		for _, e := range entries {
			statuses = append(statuses, h.checkContainerHealth(e))
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(statuses); err != nil {
			log.Printf("openclaw: statuses encode error: %v", err)
		}
	}
}

// --- Next Port ---

func (h *OpenClawHandler) NextPortHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		h.mu.RLock()
		port := h.nextAvailablePort()
		h.mu.RUnlock()
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]int{"port": port}); err != nil {
			log.Printf("openclaw: next-port encode error: %v", err)
		}
	}
}

// --- Skills ---

func (h *OpenClawHandler) SkillsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		skills, err := h.loadSkills()
		if err != nil {
			log.Printf("Warning: failed to load skills config: %v", err)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte("[]"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(skills); err != nil {
			log.Printf("openclaw: skills encode error: %v", err)
		}
	}
}

func (h *OpenClawHandler) loadSkills() ([]SkillTemplate, error) {
	candidates := []string{
		filepath.Join(h.dataDir, "skills.json"),
		filepath.Join(h.dataDir, "..", "..", "config", "openclaw-skills.json"),
	}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(exe), "config", "openclaw-skills.json"))
	}
	for _, configPath := range candidates {
		data, err := os.ReadFile(configPath)
		if err != nil {
			continue
		}
		var skills []SkillTemplate
		if err := json.Unmarshal(data, &skills); err != nil {
			return nil, fmt.Errorf("invalid %s: %w", configPath, err)
		}
		return skills, nil
	}
	return []SkillTemplate{}, nil
}

// --- Provision ---

func (h *OpenClawHandler) ProvisionHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if h.readOnly {
			http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
			return
		}

		var req ProvisionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf(`{"error":"Invalid request: %v"}`, err), http.StatusBadRequest)
			return
		}

		if req.Container.ContainerName == "" {
			req.Container.ContainerName = "openclaw-demo"
		}
		if req.Container.AuthToken == "" {
			req.Container.AuthToken = generateToken(24)
		}
		if req.Container.BaseImage == "" {
			req.Container.BaseImage = "openclaw:local"
		}
		if req.Container.NetworkMode == "" {
			req.Container.NetworkMode = "host"
		}
		if req.Container.ModelAPIKey == "" {
			req.Container.ModelAPIKey = "not-needed"
		}
		if req.Container.ModelName == "" {
			req.Container.ModelName = "auto"
		}
		if req.Container.MemoryBackend == "" {
			req.Container.MemoryBackend = "remote"
		}

		h.mu.Lock()

		if req.Container.GatewayPort == 0 {
			req.Container.GatewayPort = h.nextAvailablePort()
		} else {
			entries, _ := h.loadRegistry()
			for _, e := range entries {
				if e.Port == req.Container.GatewayPort && e.Name != req.Container.ContainerName {
					h.mu.Unlock()
					writeJSONError(w, fmt.Sprintf("Port %d already used by container %q", req.Container.GatewayPort, e.Name), http.StatusConflict)
					return
				}
			}
		}

		cDir := h.containerDataDir(req.Container.ContainerName)
		wsDir := filepath.Join(cDir, "workspace")
		for _, sub := range []string{
			"workspace",
			"workspace/memory",
			"workspace/skills",
		} {
			if err := os.MkdirAll(filepath.Join(cDir, sub), 0o755); err != nil {
				h.mu.Unlock()
				writeJSONError(w, fmt.Sprintf("Failed to create %s: %v", sub, err), http.StatusInternalServerError)
				return
			}
		}

		if err := writeIdentityFiles(wsDir, req.Identity); err != nil {
			h.mu.Unlock()
			writeJSONError(w, fmt.Sprintf("Failed to write identity files: %v", err), http.StatusInternalServerError)
			return
		}
		if err := os.WriteFile(filepath.Join(wsDir, "AGENTS.md"), []byte(agentsMdContent()), 0o644); err != nil {
			h.mu.Unlock()
			writeJSONError(w, fmt.Sprintf("Failed to write AGENTS.md: %v", err), http.StatusInternalServerError)
			return
		}

		for _, skillID := range req.Skills {
			content := h.fetchSkillContent(skillID, req.Container.BaseImage)
			if content == "" {
				continue
			}
			skillDir := filepath.Join(wsDir, "skills", skillID)
			if err := os.MkdirAll(skillDir, 0o755); err != nil {
				log.Printf("openclaw: failed to create skill dir %s: %v", skillID, err)
				continue
			}
			if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte(content), 0o644); err != nil {
				log.Printf("openclaw: failed to write skill %s: %v", skillID, err)
			}
		}

		configPath := filepath.Join(cDir, "openclaw.json")
		if err := writeOpenClawConfig(configPath, req); err != nil {
			h.mu.Unlock()
			writeJSONError(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
			return
		}

		_ = exec.Command("docker", "rm", "-f", req.Container.ContainerName).Run() // #nosec G204

		absCDir, _ := filepath.Abs(cDir)
		volumeName := "openclaw-state-" + req.Container.ContainerName
		args := []string{
			"run", "-d",
			"--name", req.Container.ContainerName,
			"--user", "0:0",
			"--network", req.Container.NetworkMode,
			"-v", absCDir + "/workspace:/workspace",
			"-v", absCDir + "/openclaw.json:/config/openclaw.json:ro",
			"-v", volumeName + ":/state",
			"-e", "OPENCLAW_CONFIG_PATH=/config/openclaw.json",
			"-e", "OPENCLAW_STATE_DIR=/state",
			req.Container.BaseImage,
			"node", "openclaw.mjs", "gateway", "--allow-unconfigured", "--bind", "lan",
		}
		out, err := exec.Command("docker", args...).CombinedOutput() // #nosec G204
		if err != nil {
			h.mu.Unlock()
			writeJSONError(w, fmt.Sprintf("Failed to start container: %s (%v)", strings.TrimSpace(string(out)), err), http.StatusInternalServerError)
			return
		}
		containerID := strings.TrimSpace(string(out))

		entries, _ := h.loadRegistry()
		found := false
		for i := range entries {
			if entries[i].Name == req.Container.ContainerName {
				entries[i].Port = req.Container.GatewayPort
				entries[i].Image = req.Container.BaseImage
				entries[i].Token = req.Container.AuthToken
				entries[i].DataDir = absCDir
				found = true
				break
			}
		}
		if !found {
			entries = append(entries, ContainerEntry{
				Name:      req.Container.ContainerName,
				Port:      req.Container.GatewayPort,
				Image:     req.Container.BaseImage,
				Token:     req.Container.AuthToken,
				DataDir:   absCDir,
				CreatedAt: time.Now().UTC().Format(time.RFC3339),
			})
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
		if err := h.saveRegistry(entries); err != nil {
			log.Printf("openclaw: failed to save registry: %v", err)
		}
		h.mu.Unlock()

		healthy := false
		client := &http.Client{Timeout: 3 * time.Second}
		gatewayURL := fmt.Sprintf("http://127.0.0.1:%d", req.Container.GatewayPort)
		for i := 0; i < 10; i++ {
			time.Sleep(2 * time.Second)
			resp, err := client.Get(gatewayURL + "/health")
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusNoContent {
					healthy = true
					break
				}
			}
		}

		msg := "Container started and gateway is healthy"
		if !healthy {
			msg = "Container started but gateway has not become healthy yet (may still be initializing)"
		}

		dockerCmd := generateDockerRunCmd(req, absCDir)
		composeYAML := generateComposeYAML(req, absCDir)

		log.Printf("OpenClaw provisioned: name=%s port=%d healthy=%v", req.Container.ContainerName, req.Container.GatewayPort, healthy)
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(ProvisionResponse{
			Success:      true,
			Message:      msg,
			WorkspaceDir: wsDir,
			ConfigPath:   configPath,
			ContainerID:  containerID,
			DockerCmd:    dockerCmd,
			ComposeYAML:  composeYAML,
		}); err != nil {
			log.Printf("openclaw: provision encode error: %v", err)
		}
	}
}

// --- Start / Stop / Delete ---

func (h *OpenClawHandler) StartHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if h.readOnly {
			http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
			return
		}
		var req struct {
			ContainerName string `json:"containerName"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, "invalid request body", http.StatusBadRequest)
			return
		}
		if req.ContainerName == "" {
			writeJSONError(w, "containerName required", http.StatusBadRequest)
			return
		}
		out, err := exec.Command("docker", "start", req.ContainerName).CombinedOutput() // #nosec G204
		if err != nil {
			writeJSONError(w, fmt.Sprintf("Failed to start: %s (%v)", strings.TrimSpace(string(out)), err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": fmt.Sprintf("Container %s started", req.ContainerName),
		}); err != nil {
			log.Printf("openclaw: start encode error: %v", err)
		}
	}
}

func (h *OpenClawHandler) StopHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if h.readOnly {
			http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
			return
		}
		var req struct {
			ContainerName string `json:"containerName"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, "invalid request body", http.StatusBadRequest)
			return
		}
		if req.ContainerName == "" {
			writeJSONError(w, "containerName required", http.StatusBadRequest)
			return
		}
		out, err := exec.Command("docker", "stop", req.ContainerName).CombinedOutput() // #nosec G204
		if err != nil {
			writeJSONError(w, fmt.Sprintf("Failed to stop: %s (%v)", strings.TrimSpace(string(out)), err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": fmt.Sprintf("Container %s stopped", req.ContainerName),
		}); err != nil {
			log.Printf("openclaw: stop encode error: %v", err)
		}
	}
}

func (h *OpenClawHandler) DeleteHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if h.readOnly {
			http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
			return
		}
		name := strings.TrimPrefix(r.URL.Path, "/api/openclaw/containers/")
		if name == "" {
			writeJSONError(w, "container name required in path", http.StatusBadRequest)
			return
		}

		_ = exec.Command("docker", "rm", "-f", name).Run() // #nosec G204

		h.mu.Lock()
		entries, _ := h.loadRegistry()
		filtered := entries[:0]
		for _, e := range entries {
			if e.Name != name {
				filtered = append(filtered, e)
			}
		}
		if err := h.saveRegistry(filtered); err != nil {
			log.Printf("openclaw: failed to save registry on delete: %v", err)
		}
		h.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": fmt.Sprintf("Container %s removed", name),
		}); err != nil {
			log.Printf("openclaw: delete encode error: %v", err)
		}
	}
}

// --- Dynamic Proxy Lookup ---

// PortForContainer returns the port for a registered container (used by dynamic proxy).
func (h *OpenClawHandler) PortForContainer(name string) (int, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	entries, err := h.loadRegistry()
	if err != nil {
		return 0, false
	}
	for _, e := range entries {
		if e.Name == name {
			return e.Port, true
		}
	}
	return 0, false
}

// --- Helpers ---

func writeJSONError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(map[string]string{"error": msg}); err != nil {
		log.Printf("openclaw: error encode error: %v", err)
	}
}

func generateToken(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "changeme-" + fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

func writeIdentityFiles(wsDir string, id IdentityConfig) error {
	var soulParts []string
	soulParts = append(soulParts, "# SOUL.md - Who You Are\n")
	if id.Name != "" || id.Role != "" {
		soulParts = append(soulParts, "## Core Identity\n")
		if id.Name != "" && id.Role != "" {
			soulParts = append(soulParts, fmt.Sprintf("You are **%s**, %s.\n", id.Name, id.Role))
		} else if id.Name != "" {
			soulParts = append(soulParts, fmt.Sprintf("You are **%s**.\n", id.Name))
		}
	}
	if id.Principles != "" {
		soulParts = append(soulParts, "## Core Truths\n")
		soulParts = append(soulParts, id.Principles+"\n")
	}
	if id.Boundaries != "" {
		soulParts = append(soulParts, "## Boundaries\n")
		soulParts = append(soulParts, id.Boundaries+"\n")
	}
	if err := os.WriteFile(filepath.Join(wsDir, "SOUL.md"), []byte(strings.Join(soulParts, "\n")), 0o644); err != nil {
		return err
	}

	var idParts []string
	idParts = append(idParts, "# IDENTITY.md - Who Am I?\n")
	if id.Name != "" {
		idParts = append(idParts, fmt.Sprintf("- **Name:** %s", id.Name))
	}
	if id.Role != "" {
		idParts = append(idParts, fmt.Sprintf("- **Creature:** %s", id.Role))
	}
	if id.Vibe != "" {
		idParts = append(idParts, fmt.Sprintf("- **Vibe:** %s", id.Vibe))
	}
	if id.Emoji != "" {
		idParts = append(idParts, fmt.Sprintf("- **Emoji:** %s", id.Emoji))
	}
	if err := os.WriteFile(filepath.Join(wsDir, "IDENTITY.md"), []byte(strings.Join(idParts, "\n")+"\n"), 0o644); err != nil {
		return err
	}

	var userParts []string
	userParts = append(userParts, "# USER.md - About Your Human\n")
	if id.UserName != "" {
		userParts = append(userParts, fmt.Sprintf("- **Name:** %s", id.UserName))
	}
	if id.UserNotes != "" {
		userParts = append(userParts, fmt.Sprintf("- **Notes:** %s", id.UserNotes))
	}
	return os.WriteFile(filepath.Join(wsDir, "USER.md"), []byte(strings.Join(userParts, "\n")+"\n"), 0o644)
}

func writeOpenClawConfig(path string, req ProvisionRequest) error {
	cfg := map[string]interface{}{
		"models": map[string]interface{}{
			"providers": map[string]interface{}{
				"vllm": map[string]interface{}{
					"baseUrl": req.Container.ModelBaseURL,
					"apiKey":  req.Container.ModelAPIKey,
					"api":     "openai-completions",
					"headers": map[string]string{"x-authz-user-id": "openclaw-demo-user"},
					"models": []map[string]interface{}{
						{
							"id": req.Container.ModelName, "name": "SR Routed Model",
							"reasoning": false, "input": []string{"text", "image"},
							"cost":          map[string]interface{}{"input": 0.15, "output": 0.6, "cacheRead": 0, "cacheWrite": 0},
							"contextWindow": 30000, "maxTokens": 1024,
							"compat": map[string]string{"maxTokensField": "max_tokens"},
						},
					},
				},
			},
		},
		"agents": map[string]interface{}{
			"defaults": map[string]interface{}{
				"model":      map[string]string{"primary": "vllm/" + req.Container.ModelName},
				"workspace":  "/workspace",
				"compaction": map[string]string{"mode": "safeguard"},
			},
			"list": []map[string]interface{}{
				{"id": "demo", "default": true, "name": "Demo Agent", "workspace": "/workspace"},
			},
		},
		"commands": map[string]interface{}{"native": "auto", "nativeSkills": "auto", "restart": true},
		"gateway": map[string]interface{}{
			"port": req.Container.GatewayPort,
			"auth": map[string]string{"mode": "token", "token": req.Container.AuthToken},
			"controlUi": map[string]interface{}{
				"dangerouslyDisableDeviceAuth": true,
				"allowInsecureAuth":            true,
				"allowedOrigins":               []string{"*"},
			},
		},
	}
	if req.Container.MemoryBackend == "remote" && req.Container.MemoryBaseURL != "" {
		cfg["memory"] = map[string]interface{}{
			"backend": "remote",
			"remote": map[string]interface{}{
				"baseUrl":              req.Container.MemoryBaseURL,
				"vectorStoreName":      req.Container.VectorStore,
				"syncIntervalMs":       30000,
				"searchMaxResults":     5,
				"searchScoreThreshold": 0.3,
			},
		}
	}
	if req.Container.BrowserEnabled {
		cfg["browser"] = map[string]interface{}{"enabled": true, "headless": true, "noSandbox": true}
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func generateDockerRunCmd(req ProvisionRequest, dataDir string) string {
	volumeName := "openclaw-state-" + req.Container.ContainerName
	return fmt.Sprintf(`docker run -d \
  --name %s \
  --user 0:0 \
  --network %s \
  -v %s/workspace:/workspace \
  -v %s/openclaw.json:/config/openclaw.json:ro \
  -v %s:/state \
  -e OPENCLAW_CONFIG_PATH=/config/openclaw.json \
  -e OPENCLAW_STATE_DIR=/state \
  %s \
  node openclaw.mjs gateway --allow-unconfigured --bind lan`,
		req.Container.ContainerName, req.Container.NetworkMode,
		dataDir, dataDir, volumeName, req.Container.BaseImage)
}

func generateComposeYAML(req ProvisionRequest, dataDir string) string {
	volumeName := "openclaw-state-" + req.Container.ContainerName
	return fmt.Sprintf(`services:
  openclaw:
    image: %s
    container_name: %s
    user: "0:0"
    network_mode: %s
    volumes:
      - %s/workspace:/workspace
      - %s/openclaw.json:/config/openclaw.json:ro
      - %s:/state
    environment:
      OPENCLAW_CONFIG_PATH: /config/openclaw.json
      OPENCLAW_STATE_DIR: /state
    command: ["node", "openclaw.mjs", "gateway", "--allow-unconfigured", "--bind", "lan"]
    restart: unless-stopped

volumes:
  %s:
`, req.Container.BaseImage, req.Container.ContainerName, req.Container.NetworkMode,
		dataDir, dataDir, volumeName, volumeName)
}

func agentsMdContent() string {
	return `# AGENTS.md - Your Workspace

This folder is home. Treat it that way.

## Every Session

Before doing anything else:

1. Read ` + "`SOUL.md`" + ` — this is who you are
2. Read ` + "`USER.md`" + ` — this is who you're helping
3. Read ` + "`memory/`" + ` for recent context

## Memory

You wake up fresh each session. These files are your continuity:

- **Daily notes:** ` + "`memory/YYYY-MM-DD.md`" + ` — raw logs of what happened
- **Skills:** ` + "`skills/*/SKILL.md`" + ` — your specialized abilities

## Safety

- Don't exfiltrate private data
- Don't run destructive commands without asking
- When in doubt, ask

## Tools

Skills provide your tools. When you need one, check its SKILL.md.
`
}

func (h *OpenClawHandler) fetchSkillContent(skillID, baseImage string) string {
	containerPaths := []string{
		"/app/skills/" + skillID + "/SKILL.md",
		"/app/extensions/" + skillID + "/SKILL.md",
	}
	for _, p := range containerPaths {
		out, err := exec.Command("docker", "run", "--rm", baseImage, "cat", p).Output() // #nosec G204
		if err == nil && len(out) > 0 {
			return string(out)
		}
	}
	skills, err := h.loadSkills()
	if err != nil {
		return ""
	}
	for _, s := range skills {
		if s.ID == skillID {
			return fmt.Sprintf("---\nname: %s\ndescription: %q\nuser-invocable: true\n---\n\n# %s\n\n%s\n",
				s.ID, s.Description, s.Name, s.Description)
		}
	}
	return ""
}
