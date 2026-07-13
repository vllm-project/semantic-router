package handlers

import (
	"context"
	"encoding/json"
	"errors"
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

	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

var containerNameInvalidChars = regexp.MustCompile(`[^a-z0-9_.-]+`)

var (
	openClawImageReferencePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]*(?:@[A-Za-z0-9_+.-]+:[A-Fa-f0-9]+)?$`)
	openClawSHA256ImagePattern    = regexp.MustCompile(`@sha256:[A-Fa-f0-9]{64}$`)
	openClawNetworkNamePattern    = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`)
)

// --- Registry ---

type ContainerEntry struct {
	Name            string `json:"name"`
	Port            int    `json:"port"`
	Image           string `json:"image"`
	Token           string `json:"-"`
	DataDir         string `json:"-"`
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

// containerEntryRecord is the private persistence envelope. ContainerEntry is
// also the public worker response type, so credentials and host filesystem
// paths must never inherit its JSON surface.
type containerEntryRecord struct {
	ContainerEntry
	Token   string `json:"token"`
	DataDir string `json:"dataDir"`
}

func newContainerEntryRecord(entry ContainerEntry) containerEntryRecord {
	return containerEntryRecord{
		ContainerEntry: entry,
		Token:          entry.Token,
		DataDir:        entry.DataDir,
	}
}

func (record containerEntryRecord) entry() ContainerEntry {
	entry := record.ContainerEntry
	entry.Token = record.Token
	entry.DataDir = record.DataDir
	return entry
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
	dataDir                   string
	readOnly                  bool
	routerConfigPath          string
	wf                        *workflowstore.Store
	mu                        sync.RWMutex
	roomWSClients             sync.Map
	roomSSEClients            sync.Map
	roomSSELastEvent          sync.Map
	roomAutomationMu          sync.Map
	roomAutomationAdmissionMu sync.Mutex
	roomAutomationAdmissions  map[string]*roomAutomationAdmission
	roomAutomationSlots       chan struct{}
	roomAutomationProcess     func(string, string)
}

// NewOpenClawHandler constructs the OpenClaw HTTP handler. wf holds durable registry,
// team, room, and message state; live SSE/WebSocket client maps stay on the handler.
func NewOpenClawHandler(dataDir string, readOnly bool, wf *workflowstore.Store) *OpenClawHandler {
	if wf == nil {
		panic("openclaw: workflow store is required")
	}
	handler := &OpenClawHandler{
		dataDir:                  dataDir,
		readOnly:                 readOnly,
		wf:                       wf,
		roomAutomationAdmissions: make(map[string]*roomAutomationAdmission),
		roomAutomationSlots:      make(chan struct{}, maximumOpenClawAutomationWorkers),
	}
	handler.roomAutomationProcess = handler.processRoomUserMessage
	return handler
}

func (h *OpenClawHandler) SetRouterConfigPath(configPath string) {
	h.routerConfigPath = strings.TrimSpace(configPath)
}

func (h *OpenClawHandler) roomMessagesPath(roomID string) string {
	return filepath.Join(h.dataDir, "room-messages", sanitizeRoomID(roomID)+".json")
}

func (h *OpenClawHandler) loadRegistry() ([]ContainerEntry, error) {
	lines, err := h.wf.ListOpenClawContainerJSON()
	if err != nil {
		return nil, err
	}
	out := make([]ContainerEntry, 0, len(lines))
	for _, line := range lines {
		var record containerEntryRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, err
		}
		out = append(out, record.entry())
	}
	return out, nil
}

func (h *OpenClawHandler) saveRegistry(entries []ContainerEntry) error {
	rows := make([][2]string, 0, len(entries))
	for i := range entries {
		b, err := json.Marshal(newContainerEntryRecord(entries[i]))
		if err != nil {
			return err
		}
		if strings.TrimSpace(entries[i].Name) == "" {
			return fmt.Errorf("container entry missing name")
		}
		rows = append(rows, [2]string{entries[i].Name, string(b)})
	}
	return h.wf.ReplaceOpenClawContainers(rows)
}

func (h *OpenClawHandler) loadTeams() ([]TeamEntry, error) {
	lines, err := h.wf.ListOpenClawTeamJSON()
	if err != nil {
		return nil, err
	}
	out := make([]TeamEntry, 0, len(lines))
	for _, line := range lines {
		var e TeamEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, nil
}

func (h *OpenClawHandler) saveTeams(entries []TeamEntry) error {
	rows := make([][2]string, 0, len(entries))
	for i := range entries {
		b, err := json.Marshal(entries[i])
		if err != nil {
			return err
		}
		if strings.TrimSpace(entries[i].ID) == "" {
			return fmt.Errorf("team entry missing id")
		}
		rows = append(rows, [2]string{entries[i].ID, string(b)})
	}
	return h.wf.ReplaceOpenClawTeams(rows)
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

// defaultBridgeGatewayPort is the fixed port used for all OpenClaw containers
// when running in bridge network mode. Since each container has its own network
// namespace with a unique IP, port conflicts cannot occur.
const defaultBridgeGatewayPort = 18790

// isBridgeNetwork returns true if the network mode is a user-defined bridge network
// (not "host" and not "container:xxx"). In bridge mode, each container has an
// isolated network namespace, so all containers can safely bind to the same port.
func isBridgeNetwork(networkMode string) bool {
	nm := strings.ToLower(strings.TrimSpace(networkMode))
	if nm == "" || nm == "host" {
		return false
	}
	if strings.HasPrefix(nm, "container:") {
		return false
	}
	return true
}

func dashboardUsesProductionSecurityProfile() bool {
	return strings.EqualFold(strings.TrimSpace(os.Getenv("DASHBOARD_SECURITY_PROFILE")), "production")
}

func validateOpenClawImageReference(image string, requireDigest bool) error {
	image = strings.TrimSpace(image)
	if image == "" || len(image) > 512 || !openClawImageReferencePattern.MatchString(image) {
		return errors.New("OpenClaw image must be one bounded OCI image reference")
	}
	if requireDigest && !openClawSHA256ImagePattern.MatchString(image) {
		return errors.New("production OpenClaw images must be pinned by sha256 digest")
	}
	return nil
}

func validateOpenClawNetworkMode(networkMode string, production bool) error {
	networkMode = strings.TrimSpace(networkMode)
	if strings.HasPrefix(strings.ToLower(networkMode), "container:") {
		name := strings.TrimSpace(strings.TrimPrefix(networkMode, "container:"))
		if name == "" || !openClawNetworkNamePattern.MatchString(name) {
			return errors.New("OpenClaw container network target is invalid")
		}
		if production {
			return errors.New("production OpenClaw workers require an isolated user-defined network")
		}
		return nil
	}
	if networkMode == "host" || networkMode == "bridge" {
		if production {
			return errors.New("production OpenClaw workers require an isolated user-defined network")
		}
		return nil
	}
	if !openClawNetworkNamePattern.MatchString(networkMode) {
		return errors.New("OpenClaw network name is invalid")
	}
	return nil
}

func (h *OpenClawHandler) nextAvailablePort(networkMode string) int {
	// In bridge network mode, all containers can safely use the same port
	// because each container has its own network namespace with a unique IP.
	if isBridgeNetwork(networkMode) {
		return defaultBridgeGatewayPort
	}

	// Host network mode: need to find an available port on the host
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
		"/usr/local/bin/docker",
		"/usr/bin/docker",
		"/bin/docker",
	}

	seen := make(map[string]bool)
	checked := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate == "" || seen[candidate] {
			continue
		}
		seen[candidate] = true
		checked = append(checked, candidate)

		if containerRuntimeLooksLikePodman(candidate) {
			return "", fmt.Errorf(
				"podman is not supported for local OpenClaw provisioning; use Docker inside the dashboard runtime",
			)
		}

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
		"container runtime not available (checked: %s). PATH=%q. OpenClaw requires Docker in the dashboard runtime. If you use `vllm-sr serve`, ensure the vllm-sr image includes the Docker CLI and mounts /var/run/docker.sock",
		strings.Join(checked, ", "), os.Getenv("PATH"),
	)
}

func containerRuntimeLooksLikePodman(candidate string) bool {
	base := strings.ToLower(filepath.Base(strings.TrimSpace(candidate)))
	return strings.HasPrefix(base, "podman")
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
	if candidate := openClawModelBaseURLFromTargetEnvoy(); candidate != "" {
		return candidate
	}
	return "http://127.0.0.1:8801/v1"
}

func defaultOpenClawModelContextWindow() int {
	if candidate := strings.TrimSpace(os.Getenv("OPENCLAW_MODEL_CONTEXT_WINDOW")); candidate != "" {
		if parsed, err := strconv.Atoi(candidate); err == nil && parsed > 0 {
			return parsed
		}
	}
	return 262144
}

func normalizeOpenClawModelContextWindow(requested int) int {
	if requested > 0 {
		return requested
	}
	return defaultOpenClawModelContextWindow()
}

func (h *OpenClawHandler) resolveOpenClawModelBaseURL() string {
	if candidate := strings.TrimSpace(os.Getenv("OPENCLAW_MODEL_BASE_URL")); candidate != "" {
		return candidate
	}
	if candidate := openClawModelBaseURLFromTargetEnvoy(); candidate != "" {
		return candidate
	}
	if candidate := h.discoverOpenClawModelBaseURLFromRouterConfig(); candidate != "" {
		return candidate
	}
	return defaultOpenClawModelBaseURL()
}

func openClawModelBaseURLFromTargetEnvoy() string {
	if candidate := strings.TrimSpace(os.Getenv("TARGET_ENVOY_URL")); candidate != "" {
		return appendOpenClawV1Path(candidate)
	}
	return ""
}

func (h *OpenClawHandler) discoverOpenClawModelBaseURLFromRouterConfig() string {
	configPath := strings.TrimSpace(h.routerConfigPath)
	if configPath == "" {
		return ""
	}

	endpoint, ok, err := routercontract.ReadFirstListenerEndpoint(configPath)
	if err != nil || !ok {
		return ""
	}

	host := formatOpenClawURLHost(normalizeOpenClawListenerHost(endpoint.Address))
	return fmt.Sprintf("http://%s:%d/v1", host, endpoint.Port)
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

func isLocalOnlyOpenClawImage(image string) bool {
	return strings.HasSuffix(strings.ToLower(strings.TrimSpace(image)), ":local")
}

func (h *OpenClawHandler) resolveBaseImage(requested string) string {
	requested = strings.TrimSpace(requested)
	if requested != "" {
		return requested
	}
	return defaultOpenClawBaseImage()
}

func (h *OpenClawHandler) ensureImageAvailable(image string) error {
	image = strings.TrimSpace(image)
	if image == "" {
		return fmt.Errorf("OpenClaw image is empty")
	}
	if isLocalOnlyOpenClawImage(image) {
		if h.imageExists(image) {
			return nil
		}
		return fmt.Errorf(
			"OpenClaw image %q is missing locally and cannot be auto-pulled. Build/tag this image locally or set OPENCLAW_BASE_IMAGE to a pullable image",
			image,
		)
	}

	out, err := h.containerCombinedOutput("pull", image)
	if err == nil {
		log.Printf("openclaw: refreshed image %q before provision", image)
		return nil
	}

	log.Printf(
		"openclaw: image refresh failed image=%q output_bytes=%d",
		image,
		len(out),
	)
	return fmt.Errorf("failed to pull OpenClaw image %q", image)
}

func (h *OpenClawHandler) runContainerCommand(args ...string) (containerCommandOutput, error) {
	runtimeBin, err := detectContainerRuntime()
	if err != nil {
		return containerCommandOutput{}, err
	}
	timeout := 30 * time.Second
	if len(args) > 0 && args[0] == "pull" {
		timeout = 10 * time.Minute
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return runBoundedCommandSplit(ctx, runtimeBin, 8*1024*1024, args...)
}

func (h *OpenClawHandler) containerOutput(args ...string) ([]byte, error) {
	output, err := h.runContainerCommand(args...)
	return output.stdout, err
}

func (h *OpenClawHandler) containerCombinedOutput(args ...string) ([]byte, error) {
	output, err := h.runContainerCommand(args...)
	combined := make([]byte, 0, len(output.stdout)+len(output.stderr))
	combined = append(combined, output.stdout...)
	combined = append(combined, output.stderr...)
	return combined, err
}

func (h *OpenClawHandler) containerRun(args ...string) error {
	_, err := h.runContainerCommand(args...)
	return err
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
	ContainerName      string `json:"containerName"`
	GatewayPort        int    `json:"gatewayPort"`
	AuthToken          string `json:"authToken"`
	ModelBaseURL       string `json:"modelBaseUrl"`
	ModelAPIKey        string `json:"modelApiKey"`
	ModelName          string `json:"modelName"`
	ModelContextWindow int    `json:"modelContextWindow,omitempty"`
	MemoryBackend      string `json:"memoryBackend"`
	MemoryBaseURL      string `json:"memoryBaseUrl"`
	VectorStore        string `json:"vectorStore"`
	BrowserEnabled     bool   `json:"browserEnabled"`
	BaseImage          string `json:"baseImage"`
	NetworkMode        string `json:"networkMode"`
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
