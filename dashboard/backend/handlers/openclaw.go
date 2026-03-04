package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

var containerNameInvalidChars = regexp.MustCompile(`[^a-z0-9_.-]+`)

// --- Registry ---

type ContainerEntry struct {
	Name            string `json:"name"`
	Port            int    `json:"port"`
	Image           string `json:"image"`
	Token           string `json:"token"`
	DataDir         string `json:"dataDir"`
	CreatedAt       string `json:"createdAt"`
	TeamID          string `json:"teamId,omitempty"`
	TeamName        string `json:"teamName,omitempty"`
	AgentName       string `json:"agentName,omitempty"`
	AgentEmoji      string `json:"agentEmoji,omitempty"`
	AgentRole       string `json:"agentRole,omitempty"`
	AgentVibe       string `json:"agentVibe,omitempty"`
	AgentPrinciples string `json:"agentPrinciples,omitempty"`
	RoleKind        string `json:"roleKind,omitempty"`
}

type TeamEntry struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Vibe        string `json:"vibe,omitempty"`
	Role        string `json:"role,omitempty"`
	Principal   string `json:"principal,omitempty"`
	Description string `json:"description,omitempty"`
	LeaderID    string `json:"leaderId,omitempty"`
	CreatedAt   string `json:"createdAt"`
	UpdatedAt   string `json:"updatedAt"`
}

type OpenClawHandler struct {
	dataDir          string
	readOnly         bool
	routerConfigPath string
	mu               sync.RWMutex
	roomSSEClients   sync.Map
	roomSSELastEvent sync.Map
	roomAutomationMu sync.Map
}

func NewOpenClawHandler(dataDir string, readOnly bool) *OpenClawHandler {
	return &OpenClawHandler{dataDir: dataDir, readOnly: readOnly}
}

func (h *OpenClawHandler) SetRouterConfigPath(configPath string) {
	h.routerConfigPath = strings.TrimSpace(configPath)
}

func (h *OpenClawHandler) registryPath() string {
	return filepath.Join(h.dataDir, "containers.json")
}

func (h *OpenClawHandler) teamsPath() string {
	return filepath.Join(h.dataDir, "teams.json")
}

func (h *OpenClawHandler) roomsPath() string {
	return filepath.Join(h.dataDir, "rooms.json")
}

func (h *OpenClawHandler) roomMessagesPath(roomID string) string {
	return filepath.Join(h.dataDir, "room-messages", sanitizeRoomID(roomID)+".json")
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

func (h *OpenClawHandler) loadTeams() ([]TeamEntry, error) {
	data, err := os.ReadFile(h.teamsPath())
	if err != nil {
		if os.IsNotExist(err) {
			return []TeamEntry{}, nil
		}
		return nil, err
	}
	var entries []TeamEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}
	return entries, nil
}

func (h *OpenClawHandler) saveTeams(entries []TeamEntry) error {
	data, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(h.teamsPath()), 0o755); err != nil {
		return err
	}
	return os.WriteFile(h.teamsPath(), data, 0o644)
}

func findTeamByID(entries []TeamEntry, id string) *TeamEntry {
	for i := range entries {
		if entries[i].ID == id {
			return &entries[i]
		}
	}
	return nil
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

func defaultOpenClawModelBaseURL() string {
	if candidate := strings.TrimSpace(os.Getenv("OPENCLAW_MODEL_BASE_URL")); candidate != "" {
		return candidate
	}
	return "http://127.0.0.1:8801/v1"
}

func (h *OpenClawHandler) resolveOpenClawModelBaseURL() string {
	if candidate := strings.TrimSpace(os.Getenv("OPENCLAW_MODEL_BASE_URL")); candidate != "" {
		return candidate
	}
	if candidate := h.discoverOpenClawModelBaseURLFromRouterConfig(); candidate != "" {
		return candidate
	}
	return defaultOpenClawModelBaseURL()
}

func (h *OpenClawHandler) discoverOpenClawModelBaseURLFromRouterConfig() string {
	configPath := strings.TrimSpace(h.routerConfigPath)
	if configPath == "" {
		return ""
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}

	var config map[string]any
	if err := yaml.Unmarshal(data, &config); err != nil {
		return ""
	}

	for _, listener := range extractOpenClawRouterListeners(config) {
		port, ok := openClawToPort(listener["port"])
		if !ok {
			continue
		}
		host := formatOpenClawURLHost(normalizeOpenClawListenerHost(asString(listener["address"])))
		return fmt.Sprintf("http://%s:%d/v1", host, port)
	}

	return ""
}

func extractOpenClawRouterListeners(config map[string]any) []map[string]any {
	listeners := make([]map[string]any, 0)

	appendListeners := func(value any) {
		entries, ok := value.([]any)
		if !ok {
			return
		}
		for _, entry := range entries {
			if listener, ok := asStringMap(entry); ok {
				listeners = append(listeners, listener)
			}
		}
	}

	appendListeners(config["listeners"])

	if apiServer, ok := asStringMap(config["api_server"]); ok {
		appendListeners(apiServer["listeners"])
	}

	return listeners
}

func asStringMap(value any) (map[string]any, bool) {
	switch typed := value.(type) {
	case map[string]any:
		return typed, true
	case map[any]any:
		normalized := make(map[string]any, len(typed))
		for key, nested := range typed {
			textKey, ok := key.(string)
			if !ok {
				continue
			}
			normalized[textKey] = nested
		}
		return normalized, true
	default:
		return nil, false
	}
}

func asString(value any) string {
	text, ok := value.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(text)
}

func openClawToPort(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		if typed >= 1 && typed <= 65535 {
			return typed, true
		}
	case int64:
		port := int(typed)
		if port >= 1 && port <= 65535 {
			return port, true
		}
	case float64:
		port := int(typed)
		if port >= 1 && port <= 65535 && float64(port) == typed {
			return port, true
		}
	case string:
		port, err := strconv.Atoi(strings.TrimSpace(typed))
		if err == nil && port >= 1 && port <= 65535 {
			return port, true
		}
	}
	return 0, false
}

func normalizeOpenClawListenerHost(host string) string {
	if host == "" || host == "0.0.0.0" || host == "::" || host == "[::]" {
		return "127.0.0.1"
	}
	return host
}

func formatOpenClawURLHost(host string) string {
	if strings.Contains(host, ":") && !strings.HasPrefix(host, "[") && !strings.HasSuffix(host, "]") {
		return "[" + host + "]"
	}
	return host
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
	TeamID    string          `json:"teamId"`
	RoleKind  string          `json:"roleKind,omitempty"`
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
	Running         bool   `json:"running"`
	ContainerName   string `json:"containerName,omitempty"`
	GatewayURL      string `json:"gatewayUrl,omitempty"`
	Port            int    `json:"port,omitempty"`
	Healthy         bool   `json:"healthy"`
	Error           string `json:"error,omitempty"`
	Image           string `json:"image,omitempty"`
	CreatedAt       string `json:"createdAt,omitempty"`
	TeamID          string `json:"teamId,omitempty"`
	TeamName        string `json:"teamName,omitempty"`
	AgentName       string `json:"agentName,omitempty"`
	AgentEmoji      string `json:"agentEmoji,omitempty"`
	AgentRole       string `json:"agentRole,omitempty"`
	AgentVibe       string `json:"agentVibe,omitempty"`
	AgentPrinciples string `json:"agentPrinciples,omitempty"`
	RoleKind        string `json:"roleKind,omitempty"`
}

type identitySnapshot struct {
	Name       string
	Emoji      string
	Role       string
	Vibe       string
	Principles string
}
