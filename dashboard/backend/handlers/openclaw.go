package handlers

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

var containerNameInvalidChars = regexp.MustCompile(`[^a-z0-9_.-]+`)

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
		if !used[port] && isTCPPortAvailable(port) {
			return port
		}
	}
}

func isTCPPortAvailable(port int) bool {
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return false
	}
	_ = ln.Close()
	return true
}

func canConnectTCP(host string, port int, timeout time.Duration) bool {
	addr := net.JoinHostPort(host, fmt.Sprintf("%d", port))
	conn, err := net.DialTimeout("tcp", addr, timeout)
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}

func detectContainerRuntime() (string, error) {
	candidates := []string{
		strings.TrimSpace(os.Getenv("OPENCLAW_CONTAINER_RUNTIME")),
		strings.TrimSpace(os.Getenv("CONTAINER_RUNTIME")),
		"docker",
		"podman",
		"/usr/local/bin/docker",
		"/usr/bin/docker",
		"/bin/docker",
		"/usr/local/bin/podman",
		"/usr/bin/podman",
		"/bin/podman",
	}

	seen := make(map[string]bool)
	checked := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate == "" || seen[candidate] {
			continue
		}
		seen[candidate] = true
		checked = append(checked, candidate)

		if filepath.IsAbs(candidate) {
			info, err := os.Stat(candidate)
			if err == nil && !info.IsDir() {
				return candidate, nil
			}
			continue
		}

		if resolved, err := exec.LookPath(candidate); err == nil {
			return resolved, nil
		}
	}

	return "", fmt.Errorf(
		"container runtime not available (checked: %s). PATH=%q. OpenClaw requires docker/podman in dashboard runtime. If you use `vllm-sr serve`, ensure vllm-sr image includes Docker CLI and mount /var/run/docker.sock",
		strings.Join(checked, ", "), os.Getenv("PATH"),
	)
}

func defaultOpenClawBaseImage() string {
	if candidate := strings.TrimSpace(os.Getenv("OPENCLAW_BASE_IMAGE")); candidate != "" {
		return candidate
	}
	return "ghcr.io/openclaw/openclaw:latest"
}

func isContainerImageMissingError(output string) bool {
	lower := strings.ToLower(output)
	return strings.Contains(lower, "unable to find image") ||
		strings.Contains(lower, "pull access denied") ||
		strings.Contains(lower, "manifest unknown") ||
		strings.Contains(lower, "repository does not exist")
}

func (h *OpenClawHandler) imageExists(image string) bool {
	if strings.TrimSpace(image) == "" {
		return false
	}
	_, err := h.containerCombinedOutput("image", "inspect", image)
	return err == nil
}

func (h *OpenClawHandler) discoverLocalOpenClawImage() string {
	out, err := h.containerOutput("image", "ls", "--format", "{{.Repository}}:{{.Tag}}")
	if err != nil {
		return ""
	}

	seen := make(map[string]bool)
	latestCandidates := make([]string, 0)
	otherCandidates := make([]string, 0)
	for _, raw := range strings.Split(string(out), "\n") {
		image := strings.TrimSpace(raw)
		if image == "" || seen[image] {
			continue
		}
		seen[image] = true

		lower := strings.ToLower(image)
		if strings.Contains(lower, "<none>") {
			continue
		}
		if !strings.Contains(lower, "openclaw") {
			continue
		}
		if strings.HasSuffix(lower, ":latest") {
			latestCandidates = append(latestCandidates, image)
		} else {
			otherCandidates = append(otherCandidates, image)
		}
	}

	if len(latestCandidates) > 0 {
		return latestCandidates[0]
	}
	if len(otherCandidates) > 0 {
		return otherCandidates[0]
	}
	return ""
}

func (h *OpenClawHandler) resolveBaseImage(requested string) string {
	requested = strings.TrimSpace(requested)
	if requested != "" && requested != "ghcr.io/openclaw/openclaw:latest" {
		return requested
	}

	configured := defaultOpenClawBaseImage()
	if configured != "ghcr.io/openclaw/openclaw:latest" {
		return configured
	}

	if h.imageExists("ghcr.io/openclaw/openclaw:latest") {
		return "ghcr.io/openclaw/openclaw:latest"
	}

	discovered := h.discoverLocalOpenClawImage()
	if discovered != "" {
		log.Printf("openclaw: auto-selected local image %q (ghcr.io/openclaw/openclaw:latest missing)", discovered)
		return discovered
	}

	return "ghcr.io/openclaw/openclaw:latest"
}

func (h *OpenClawHandler) ensureImageAvailable(image string) error {
	image = strings.TrimSpace(image)
	if image == "" {
		return fmt.Errorf("OpenClaw image is empty")
	}
	if h.imageExists(image) {
		return nil
	}

	out, err := h.containerCombinedOutput("pull", image)
	if err == nil {
		log.Printf("openclaw: pulled missing image %q", image)
		return nil
	}

	trimmed := strings.TrimSpace(string(out))
	if strings.HasSuffix(strings.ToLower(image), ":local") {
		return fmt.Errorf(
			"OpenClaw image %q is missing locally and cannot be auto-pulled. Build/tag this image locally or set OPENCLAW_BASE_IMAGE to a pullable image",
			image,
		)
	}
	if trimmed == "" {
		return fmt.Errorf("failed to pull OpenClaw image %q", image)
	}
	return fmt.Errorf("failed to pull OpenClaw image %q: %s", image, trimmed)
}

func (h *OpenClawHandler) containerCommand(args ...string) (*exec.Cmd, error) {
	runtimeBin, err := detectContainerRuntime()
	if err != nil {
		return nil, err
	}
	return exec.Command(runtimeBin, args...), nil // #nosec G204
}

func (h *OpenClawHandler) containerOutput(args ...string) ([]byte, error) {
	cmd, err := h.containerCommand(args...)
	if err != nil {
		return nil, err
	}
	return cmd.Output()
}

func (h *OpenClawHandler) containerCombinedOutput(args ...string) ([]byte, error) {
	cmd, err := h.containerCommand(args...)
	if err != nil {
		return nil, err
	}
	return cmd.CombinedOutput()
}

func (h *OpenClawHandler) containerRun(args ...string) error {
	cmd, err := h.containerCommand(args...)
	if err != nil {
		return err
	}
	return cmd.Run()
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

func (h *OpenClawHandler) gatewayHostCandidates() []string {
	candidates := []string{}

	if explicit := strings.TrimSpace(os.Getenv("OPENCLAW_GATEWAY_HOST")); explicit != "" {
		candidates = append(candidates, explicit)
	}
	if explicitList := strings.TrimSpace(os.Getenv("OPENCLAW_GATEWAY_HOSTS")); explicitList != "" {
		for _, raw := range strings.Split(explicitList, ",") {
			if host := strings.TrimSpace(raw); host != "" {
				candidates = append(candidates, host)
			}
		}
	}

	candidates = append(candidates,
		"127.0.0.1",
		"host.docker.internal",
		"host.containers.internal",
	)

	seen := map[string]bool{}
	out := make([]string, 0, len(candidates))
	for _, host := range candidates {
		if host == "" || seen[host] {
			continue
		}
		seen[host] = true
		out = append(out, host)
	}
	return out
}

func (h *OpenClawHandler) resolveGatewayHost(port int) string {
	candidates := h.gatewayHostCandidates()
	if len(candidates) == 0 {
		return "127.0.0.1"
	}
	for _, host := range candidates {
		if canConnectTCP(host, port, 350*time.Millisecond) {
			return host
		}
	}
	return candidates[0]
}

func (h *OpenClawHandler) gatewayBaseURL(port int) string {
	host := h.resolveGatewayHost(port)
	return fmt.Sprintf("http://%s:%d", host, port)
}

func (h *OpenClawHandler) gatewayReachable(port int) bool {
	client := &http.Client{Timeout: 1200 * time.Millisecond}
	for _, host := range h.gatewayHostCandidates() {
		target := fmt.Sprintf("http://%s:%d/health", host, port)
		resp, err := client.Get(target)
		if err == nil {
			resp.Body.Close()
			// Any HTTP response confirms the gateway endpoint is reachable.
			return true
		}
		if canConnectTCP(host, port, 350*time.Millisecond) {
			return true
		}
	}
	return false
}

func (h *OpenClawHandler) checkContainerHealth(entry ContainerEntry) OpenClawStatus {
	status := OpenClawStatus{
		ContainerName: entry.Name,
		GatewayURL:    h.gatewayBaseURL(entry.Port),
		Port:          entry.Port,
	}

	out, err := h.containerOutput("inspect", "-f", "{{.State.Running}}", entry.Name)
	if err != nil {
		status.Running = false
		if strings.Contains(err.Error(), "container runtime not available") {
			status.Error = err.Error()
			return status
		}
		status.Error = "Container not found"
		return status
	}
	status.Running = strings.TrimSpace(string(out)) == "true"
	if !status.Running {
		status.Error = "Container stopped"
		return status
	}

	gatewayReachable := h.gatewayReachable(entry.Port)
	if !gatewayReachable {
		status.Error = "Gateway not reachable"
		return status
	}

	// Compare positions so a successful restart after a previous failure is correctly detected.
	logOut, logErr := h.containerCombinedOutput("logs", "--tail", "80", entry.Name)
	if logErr == nil {
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
	candidates := make([]string, 0, 12)
	if p := strings.TrimSpace(os.Getenv("OPENCLAW_SKILLS_PATH")); p != "" {
		candidates = append(candidates, p)
	}

	candidates = append(candidates,
		filepath.Join(h.dataDir, "skills.json"),
		filepath.Join(h.dataDir, "..", "..", "config", "openclaw-skills.json"),
		"/app/config/openclaw-skills.json",
		"/app/dashboard/backend/config/openclaw-skills.json",
		"./config/openclaw-skills.json",
	)

	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(wd, "config", "openclaw-skills.json"))
	}
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		candidates = append(candidates,
			filepath.Join(exeDir, "config", "openclaw-skills.json"),
			filepath.Join(exeDir, "..", "config", "openclaw-skills.json"),
		)
	}

	seen := make(map[string]struct{}, len(candidates))
	for _, rawPath := range candidates {
		configPath := strings.TrimSpace(rawPath)
		if configPath == "" {
			continue
		}
		cleanPath := filepath.Clean(configPath)
		if _, ok := seen[cleanPath]; ok {
			continue
		}
		seen[cleanPath] = struct{}{}

		data, err := os.ReadFile(configPath)
		if err != nil {
			continue
		}
		var skills []SkillTemplate
		if err := json.Unmarshal(data, &skills); err != nil {
			return nil, fmt.Errorf("invalid %s: %w", configPath, err)
		}
		log.Printf("openclaw: loaded %d skills from %s", len(skills), configPath)
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

		runtimeBin, runtimeErr := detectContainerRuntime()
		if runtimeErr != nil {
			writeJSONError(w, runtimeErr.Error(), http.StatusServiceUnavailable)
			return
		}
		runtimeName := filepath.Base(runtimeBin)
		if runtimeName == "" {
			runtimeName = runtimeBin
		}

		var req ProvisionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf(`{"error":"Invalid request: %v"}`, err), http.StatusBadRequest)
			return
		}

		req.Container.ContainerName = deriveContainerName(req.Container.ContainerName, req.Identity.Name)
		if req.Container.AuthToken == "" {
			req.Container.AuthToken = generateToken(24)
		}
		req.Container.BaseImage = h.resolveBaseImage(req.Container.BaseImage)
		if preferredNetwork := strings.TrimSpace(os.Getenv("OPENCLAW_DEFAULT_NETWORK_MODE")); preferredNetwork != "" {
			// In vllm-sr serve deployment, dashboard often runs in a container while OpenClaw
			// is launched via host docker.sock. Using container:<dashboard-container> keeps
			// gateway traffic in the same network namespace and avoids host routing issues.
			if req.Container.NetworkMode == "" || strings.EqualFold(req.Container.NetworkMode, "host") {
				req.Container.NetworkMode = preferredNetwork
			}
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
			req.Container.MemoryBackend = "local"
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
			if !isTCPPortAvailable(req.Container.GatewayPort) {
				h.mu.Unlock()
				writeJSONError(
					w,
					fmt.Sprintf(
						"Port %d is already in use on host. Stop the existing gateway/container (e.g. `openclaw gateway stop`) or choose another port.",
						req.Container.GatewayPort,
					),
					http.StatusConflict,
				)
				return
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

		if err := h.ensureImageAvailable(req.Container.BaseImage); err != nil {
			h.mu.Unlock()
			writeJSONError(w, err.Error(), http.StatusBadRequest)
			return
		}

		_ = h.containerRun("rm", "-f", req.Container.ContainerName)

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
		out, err := h.containerCombinedOutput(args...)
		if err != nil {
			trimmed := strings.TrimSpace(string(out))
			if isContainerImageMissingError(trimmed) {
				h.mu.Unlock()
				writeJSONError(
					w,
					fmt.Sprintf(
						"OpenClaw image %q is unavailable on host runtime. Build or pull this image first, or set OPENCLAW_BASE_IMAGE to an available image before starting dashboard.",
						req.Container.BaseImage,
					),
					http.StatusBadRequest,
				)
				return
			}

			h.mu.Unlock()
			writeJSONError(w, fmt.Sprintf("Failed to start container: %s (%v)", trimmed, err), http.StatusInternalServerError)
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
		for i := 0; i < 10; i++ {
			time.Sleep(2 * time.Second)
			if h.gatewayReachable(req.Container.GatewayPort) {
				healthy = true
				break
			}
		}

		msg := "Container started and gateway is healthy"
		if !healthy {
			msg = "Container started but gateway has not become healthy yet (may still be initializing)"
		}

		dockerCmd := generateDockerRunCmd(runtimeName, req, absCDir)
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
		out, err := h.containerCombinedOutput("start", req.ContainerName)
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
		out, err := h.containerCombinedOutput("stop", req.ContainerName)
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

		_ = h.containerRun("rm", "-f", name)

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

// TargetBaseForContainer resolves the HTTP base URL for a registered container.
func (h *OpenClawHandler) TargetBaseForContainer(name string) (string, bool) {
	port, ok := h.PortForContainer(name)
	if !ok {
		return "", false
	}
	return h.gatewayBaseURL(port), true
}

// --- Helpers ---

func sanitizeContainerName(raw string) string {
	cleaned := strings.ToLower(strings.TrimSpace(raw))
	cleaned = containerNameInvalidChars.ReplaceAllString(cleaned, "-")
	cleaned = strings.Trim(cleaned, "._-")
	if cleaned == "" {
		return ""
	}

	// Keep names bounded and still docker-friendly.
	const maxLen = 63
	if len(cleaned) > maxLen {
		cleaned = strings.Trim(cleaned[:maxLen], "._-")
	}
	if cleaned == "" {
		return ""
	}

	first := cleaned[0]
	if (first < 'a' || first > 'z') && (first < '0' || first > '9') {
		cleaned = "oc-" + cleaned
	}
	return cleaned
}

func deriveContainerName(requested, identityName string) string {
	if name := sanitizeContainerName(requested); name != "" {
		return name
	}
	if name := sanitizeContainerName(identityName); name != "" {
		return name
	}
	return "openclaw-vllm-sr"
}

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
	if id.Vibe != "" {
		soulParts = append(soulParts, "## Vibe\n")
		soulParts = append(soulParts, id.Vibe+"\n")
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
	// Recover from stale state where a previous bad bind mount caused
	// openclaw.json to be created as a directory on host.
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		if removeErr := os.RemoveAll(path); removeErr != nil {
			return fmt.Errorf("failed to replace config directory %s with file: %w", path, removeErr)
		}
	} else if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to stat config path %s: %w", path, err)
	}

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
				{"id": "vllm-sr", "default": true, "name": "vLLM-SR Powered Agent", "workspace": "/workspace"},
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
	memoryBackend := strings.ToLower(strings.TrimSpace(req.Container.MemoryBackend))
	if memoryBackend == "" {
		memoryBackend = "local"
	}

	// OpenClaw v2 memory schema:
	// - memory.backend accepts "builtin" or "qmd"
	// - remote embedding config lives under agents.defaults.memorySearch
	switch memoryBackend {
	case "qmd":
		cfg["memory"] = map[string]interface{}{"backend": "qmd"}
	case "remote":
		cfg["memory"] = map[string]interface{}{"backend": "builtin"}

		memorySearch := map[string]interface{}{
			"enabled":  true,
			"provider": "openai",
		}

		remote := map[string]interface{}{}
		if baseURL := strings.TrimSpace(req.Container.MemoryBaseURL); baseURL != "" {
			remote["baseUrl"] = baseURL
		}
		if apiKey := strings.TrimSpace(req.Container.ModelAPIKey); apiKey != "" && apiKey != "not-needed" {
			remote["apiKey"] = apiKey
		}
		if len(remote) > 0 {
			memorySearch["remote"] = remote
		}

		agentsCfg, _ := cfg["agents"].(map[string]interface{})
		defaultsCfg, _ := agentsCfg["defaults"].(map[string]interface{})
		defaultsCfg["memorySearch"] = memorySearch
	default:
		// "local" (or unknown values) falls back to builtin memory without
		// remote embedding configuration.
		cfg["memory"] = map[string]interface{}{"backend": "builtin"}
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

func generateDockerRunCmd(runtime string, req ProvisionRequest, dataDir string) string {
	volumeName := "openclaw-state-" + req.Container.ContainerName
	return fmt.Sprintf(`%s run -d \
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
		runtime, req.Container.ContainerName, req.Container.NetworkMode,
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
2. Read ` + "`IDENTITY.md`" + ` — your profile, vibe, and persona details
3. Read ` + "`USER.md`" + ` — this is who you're helping
4. Read ` + "`memory/`" + ` for recent context

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
		out, err := h.containerOutput("run", "--rm", baseImage, "cat", p)
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
