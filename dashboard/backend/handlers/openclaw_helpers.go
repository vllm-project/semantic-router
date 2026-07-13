package handlers

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
)

const maximumOpenClawIdentityFileBytes = 64 * 1024

// rewriteLoopbackHost replaces 127.0.0.1 / localhost in a URL with the given
// container name so that inter-container traffic uses Docker DNS instead of
// loopback (which is unreachable across containers in bridge networks).
func rewriteLoopbackHost(rawURL, containerName string) string {
	if rawURL == "" || containerName == "" {
		return rawURL
	}
	u, err := url.Parse(rawURL)
	if err != nil {
		return rawURL
	}
	host := u.Hostname()
	if host != "127.0.0.1" && host != "localhost" && host != "0.0.0.0" {
		return rawURL
	}
	port := u.Port()
	if port != "" {
		u.Host = containerName + ":" + port
	} else {
		u.Host = containerName
	}
	return u.String()
}

func appendOpenClawV1Path(rawURL string) string {
	if rawURL == "" {
		return rawURL
	}

	u, err := url.Parse(rawURL)
	if err != nil {
		return rawURL
	}

	trimmedPath := strings.TrimRight(u.Path, "/")
	if trimmedPath == "/v1" {
		u.Path = "/v1"
		return u.String()
	}
	if trimmedPath == "" {
		u.Path = "/v1"
		return u.String()
	}

	u.Path = trimmedPath + "/v1"
	return u.String()
}

func openClawModelGatewayContainerName() string {
	if candidate := strings.TrimSpace(os.Getenv("OPENCLAW_MODEL_GATEWAY_CONTAINER_NAME")); candidate != "" {
		return candidate
	}
	if candidate := openClawContainerHostFromURL(strings.TrimSpace(os.Getenv("TARGET_ENVOY_URL"))); candidate != "" {
		return candidate
	}
	if candidate := strings.TrimSpace(os.Getenv(envoyContainerNameEnv)); candidate != "" {
		return candidate
	}
	if candidate := strings.TrimSpace(os.Getenv("OPENCLAW_DASHBOARD_CONTAINER_NAME")); candidate != "" {
		return candidate
	}
	return defaultRouterContainerName
}

func openClawContainerHostFromURL(rawURL string) string {
	if rawURL == "" {
		return ""
	}

	u, err := url.Parse(rawURL)
	if err != nil {
		return ""
	}

	host := strings.TrimSpace(u.Hostname())
	if host == "" || host == "127.0.0.1" || host == "localhost" || host == "0.0.0.0" {
		return ""
	}
	return host
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

func sanitizeTeamID(raw string) string {
	return sanitizeContainerName(raw)
}

func sanitizeRoomID(raw string) string {
	return sanitizeContainerName(raw)
}

func normalizeRoleKind(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "leader":
		return "leader"
	default:
		return "worker"
	}
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

func readIdentitySnapshot(dataDir string) identitySnapshot {
	wsDir := filepath.Join(dataDir, "workspace")
	snapshot := identitySnapshot{}

	identityContent, err := readPrivateOpenClawFile(
		filepath.Join(wsDir, "IDENTITY.md"),
		maximumOpenClawIdentityFileBytes,
	)
	if err == nil {
		for _, raw := range strings.Split(string(identityContent), "\n") {
			line := strings.TrimSpace(raw)
			switch {
			case strings.HasPrefix(line, "- **Name:**"):
				snapshot.Name = strings.TrimSpace(strings.TrimPrefix(line, "- **Name:**"))
			case strings.HasPrefix(line, "- **Creature:**"):
				snapshot.Role = strings.TrimSpace(strings.TrimPrefix(line, "- **Creature:**"))
			case strings.HasPrefix(line, "- **Vibe:**"):
				snapshot.Vibe = strings.TrimSpace(strings.TrimPrefix(line, "- **Vibe:**"))
			case strings.HasPrefix(line, "- **Emoji:**"):
				snapshot.Emoji = strings.TrimSpace(strings.TrimPrefix(line, "- **Emoji:**"))
			}
		}
	}

	soulContent, err := readPrivateOpenClawFile(
		filepath.Join(wsDir, "SOUL.md"),
		maximumOpenClawIdentityFileBytes,
	)
	if err == nil {
		lines := strings.Split(string(soulContent), "\n")
		capture := false
		var truths []string
		for _, raw := range lines {
			line := strings.TrimSpace(raw)
			if strings.HasPrefix(line, "## ") {
				if line == "## Core Truths" {
					capture = true
					continue
				}
				if capture {
					break
				}
			}
			if capture && line != "" {
				truths = append(truths, line)
			}
		}
		if len(truths) > 0 {
			snapshot.Principles = strings.Join(truths, " ")
		}
	}

	return snapshot
}

func writeJSONError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(map[string]string{"error": msg}); err != nil {
		log.Printf("openclaw: error encode error: %v", err)
	}
}

var openClawFallbackIDCounter atomic.Uint64

func generateRandomHex(reader io.Reader, n int) (string, error) {
	if reader == nil || n <= 0 {
		return "", errors.New("random token size and source are required")
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(reader, b); err != nil {
		return "", fmt.Errorf("read cryptographic randomness: %w", err)
	}
	return hex.EncodeToString(b), nil
}

func generateSecretToken(n int) (string, error) {
	return generateRandomHex(rand.Reader, n)
}

func generateNonSecretToken(n int) string {
	if token, err := generateSecretToken(n); err == nil {
		return token
	}
	seed := fmt.Sprintf(
		"%d:%d",
		time.Now().UTC().UnixNano(),
		openClawFallbackIDCounter.Add(1),
	)
	digest := sha256.Sum256([]byte(seed))
	encoded := hex.EncodeToString(digest[:])
	length := n * 2
	if length > len(encoded) {
		length = len(encoded)
	}
	return encoded[:length]
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
	if err := writePrivateOpenClawFile(filepath.Join(wsDir, "SOUL.md"), []byte(strings.Join(soulParts, "\n"))); err != nil {
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
	if err := writePrivateOpenClawFile(filepath.Join(wsDir, "IDENTITY.md"), []byte(strings.Join(idParts, "\n")+"\n")); err != nil {
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
	return writePrivateOpenClawFile(filepath.Join(wsDir, "USER.md"), []byte(strings.Join(userParts, "\n")+"\n"))
}

func ensureOpenClawDirectory(path string, mode os.FileMode) error {
	info, err := os.Lstat(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		if mkdirErr := os.Mkdir(path, mode); mkdirErr != nil {
			return fmt.Errorf("create private OpenClaw directory: %w", mkdirErr)
		}
	case err != nil:
		return fmt.Errorf("inspect private OpenClaw directory: %w", err)
	case info.Mode()&os.ModeSymlink != 0 || !info.IsDir():
		return errors.New("private OpenClaw directory path must not be a symlink or special file")
	}
	if err := os.Chmod(path, mode); err != nil {
		return fmt.Errorf("secure private OpenClaw directory: %w", err)
	}
	return nil
}

func buildOpenClawModelProviderConfig(req ProvisionRequest) map[string]interface{} {
	return map[string]interface{}{
		"baseUrl": req.Container.ModelBaseURL,
		"apiKey":  req.Container.ModelAPIKey,
		"api":     "openai-completions",
		"headers": map[string]string{"x-authz-user-id": "openclaw-demo-user"},
		"models": []map[string]interface{}{
			{
				"id": req.Container.ModelName, "name": "SR Routed Model",
				"reasoning": false, "input": []string{"text", "image"},
				"cost":          map[string]interface{}{"input": 0.15, "output": 0.6, "cacheRead": 0, "cacheWrite": 0},
				"contextWindow": normalizeOpenClawModelContextWindow(req.Container.ModelContextWindow),
				"compat":        map[string]string{"maxTokensField": "max_tokens"},
			},
		},
	}
}

func writeOpenClawConfig(path string, req ProvisionRequest) error {
	return writeOpenClawConfigForProfile(path, req, dashboardUsesProductionSecurityProfile())
}

func writeOpenClawConfigForProfile(path string, req ProvisionRequest, production bool) error {
	// Recover from stale state where a previous bad bind mount caused
	// openclaw.json to be created as a directory on host.
	if info, err := os.Lstat(path); err == nil && info.IsDir() {
		if removeErr := os.RemoveAll(path); removeErr != nil {
			return fmt.Errorf("failed to replace config directory %s with file: %w", path, removeErr)
		}
	} else if err == nil && (info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular()) {
		return errors.New("OpenClaw config path must be a regular file, not a symlink or special file")
	} else if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to stat config path %s: %w", path, err)
	}

	gatewayConfig := map[string]interface{}{
		"port": req.Container.GatewayPort,
		"auth": map[string]string{"mode": "token", "token": req.Container.AuthToken},
		"http": map[string]interface{}{
			"endpoints": map[string]interface{}{
				"chatCompletions": map[string]interface{}{"enabled": true},
				"responses":       map[string]interface{}{"enabled": true},
			},
		},
	}
	if !production {
		// Same-origin control UI is a trusted-development feature only. Production
		// disables its proxy boundary and does not emit the upstream insecure-auth
		// escape hatches or a wildcard browser origin.
		gatewayConfig["controlUi"] = map[string]interface{}{
			"dangerouslyDisableDeviceAuth": true,
			"allowInsecureAuth":            true,
			"allowedOrigins":               []string{"*"},
		}
	}

	cfg := map[string]interface{}{
		"models": map[string]interface{}{
			"providers": map[string]interface{}{
				"vllm": buildOpenClawModelProviderConfig(req),
			},
		},
		"agents": map[string]interface{}{
			"defaults": map[string]interface{}{
				"model":      map[string]string{"primary": "vllm/" + req.Container.ModelName},
				"workspace":  "/workspace",
				"compaction": map[string]string{"mode": "safeguard"},
			},
			"list": []map[string]interface{}{
				{"id": openClawPrimaryAgentID, "default": true, "name": "vLLM-SR Powered Agent", "workspace": "/workspace"},
			},
		},
		"commands": map[string]interface{}{"native": "auto", "nativeSkills": "auto", "restart": true},
		"gateway":  gatewayConfig,
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
	if req.Container.BrowserEnabled && !production {
		cfg["browser"] = map[string]interface{}{"enabled": true, "headless": true, "noSandbox": true}
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return writePrivateOpenClawFile(path, data)
}

func writePrivateOpenClawFile(path string, data []byte) error {
	flags := os.O_WRONLY
	created := false
	var expected os.FileInfo
	info, err := os.Lstat(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		flags = os.O_WRONLY | os.O_CREATE | os.O_EXCL
		created = true
	case err != nil:
		return fmt.Errorf("inspect private OpenClaw file: %w", err)
	case info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular():
		return errors.New("private OpenClaw path must be a regular file, not a symlink or special file")
	default:
		expected = info
	}

	file, err := os.OpenFile(path, flags, 0o600)
	if err != nil {
		return fmt.Errorf("open private OpenClaw file: %w", err)
	}
	closed := false
	succeeded := false
	defer func() {
		if !closed {
			_ = file.Close()
		}
		if created && !succeeded {
			_ = os.Remove(path)
		}
	}()
	actual, statErr := file.Stat()
	if statErr != nil {
		return fmt.Errorf("inspect opened private OpenClaw file: %w", statErr)
	}
	if !actual.Mode().IsRegular() || (expected != nil && !os.SameFile(expected, actual)) {
		return errors.New("private OpenClaw file changed during secure open")
	}
	if chmodErr := file.Chmod(0o600); chmodErr != nil {
		return fmt.Errorf("secure private OpenClaw file: %w", chmodErr)
	}
	if truncateErr := file.Truncate(0); truncateErr != nil {
		return fmt.Errorf("truncate private OpenClaw file: %w", truncateErr)
	}
	if _, seekErr := file.Seek(0, io.SeekStart); seekErr != nil {
		return fmt.Errorf("rewind private OpenClaw file: %w", seekErr)
	}
	written, err := file.Write(data)
	if err != nil {
		return fmt.Errorf("write private OpenClaw file: %w", err)
	}
	if written != len(data) {
		return io.ErrShortWrite
	}
	if syncErr := file.Sync(); syncErr != nil {
		return fmt.Errorf("sync private OpenClaw file: %w", syncErr)
	}
	if closeErr := file.Close(); closeErr != nil {
		return fmt.Errorf("close private OpenClaw file: %w", closeErr)
	}
	closed = true
	succeeded = true
	return nil
}

func readPrivateOpenClawFile(path string, maximumBytes int64) ([]byte, error) {
	if maximumBytes <= 0 {
		return nil, errors.New("private OpenClaw read limit must be positive")
	}
	expected, err := os.Lstat(path)
	if err != nil {
		return nil, fmt.Errorf("inspect private OpenClaw file: %w", err)
	}
	if expected.Mode()&os.ModeSymlink != 0 || !expected.Mode().IsRegular() {
		return nil, errors.New("private OpenClaw path must be a regular file, not a symlink or special file")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open private OpenClaw file: %w", err)
	}
	defer func() { _ = file.Close() }()
	actual, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("inspect opened private OpenClaw file: %w", err)
	}
	if !actual.Mode().IsRegular() || !os.SameFile(expected, actual) {
		return nil, errors.New("private OpenClaw file changed during secure open")
	}
	data, err := io.ReadAll(io.LimitReader(file, maximumBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read private OpenClaw file: %w", err)
	}
	if int64(len(data)) > maximumBytes {
		return nil, errors.New("private OpenClaw file exceeds read limit")
	}
	return data, nil
}

func openClawContainerRunArgs(req ProvisionRequest, dataDir, ownerID string) []string {
	volumeName := "openclaw-state-" + req.Container.ContainerName
	healthCmd := fmt.Sprintf(
		`node -e "fetch('http://127.0.0.1:%d/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"`,
		req.Container.GatewayPort,
	)
	return []string{
		"run", "-d",
		"--name", req.Container.ContainerName,
		"--label", openClawManagedLabel,
		"--label", openClawOwnerLabelKey + "=" + ownerID,
		"--user", "0:0",
		"--cap-drop", "ALL",
		"--security-opt", "no-new-privileges:true",
		"--pids-limit", "512",
		"--memory", "4g",
		"--cpus", "2.0",
		"--network", req.Container.NetworkMode,
		"--health-cmd", healthCmd,
		"--health-interval", "30s",
		"--health-timeout", "5s",
		"--health-start-period", "15s",
		"--health-retries", "3",
		"-v", dataDir + "/workspace:/workspace",
		"-v", dataDir + "/openclaw.json:/config/openclaw.json:ro",
		"-v", volumeName + ":/state",
		"-e", "OPENCLAW_CONFIG_PATH=/config/openclaw.json",
		"-e", "OPENCLAW_STATE_DIR=/state",
		req.Container.BaseImage,
		"node", "openclaw.mjs", "gateway", "--allow-unconfigured", "--bind", "lan",
	}
}

func openClawShellQuote(value string) string {
	if value != "" {
		safe := true
		for _, character := range value {
			if (character >= 'a' && character <= 'z') ||
				(character >= 'A' && character <= 'Z') ||
				(character >= '0' && character <= '9') ||
				strings.ContainsRune("_@%+=:,./-", character) {
				continue
			}
			safe = false
			break
		}
		if safe {
			return value
		}
	}
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}

func formatOpenClawShellCommand(argv []string) string {
	quoted := make([]string, len(argv))
	for index := range argv {
		quoted[index] = openClawShellQuote(argv[index])
	}
	return strings.Join(quoted, " ")
}

func generateDockerRunCmd(runtime string, req ProvisionRequest, dataDir, ownerID string) string {
	volumeName := "openclaw-state-" + req.Container.ContainerName
	volumeCommand := []string{
		runtime, "volume", "create",
		"--label", openClawManagedLabel,
		"--label", openClawOwnerLabelKey + "=" + ownerID,
		volumeName,
	}
	volumeInspectCommand := []string{
		runtime, "volume", "inspect", "--format",
		fmt.Sprintf(
			`{{with .Labels}}{{index . %q}}{{end}}|{{with .Labels}}{{index . %q}}{{end}}`,
			openClawManagedLabelKey,
			openClawOwnerLabelKey,
		),
		volumeName,
	}
	runCommand := append([]string{runtime}, openClawContainerRunArgs(req, dataDir, ownerID)...)
	return formatOpenClawShellCommand(volumeCommand) + " >/dev/null && \\\n  " +
		formatOpenClawShellCommand(volumeInspectCommand) + " | grep -Fqx -- " +
		openClawShellQuote("true|"+ownerID) + " && \\\n  " +
		formatOpenClawShellCommand(runCommand)
}

func openClawProvisionGuidance(
	production bool,
	runtime string,
	req ProvisionRequest,
	dataDir string,
	ownerID string,
) (string, string) {
	if production {
		// Generated shell/YAML is convenience output, not an owned deployment
		// artifact. Never offer copy/paste commands on the production boundary.
		return "", ""
	}
	return generateDockerRunCmd(runtime, req, dataDir, ownerID), generateComposeYAML(req, dataDir, ownerID)
}

func generateComposeYAML(req ProvisionRequest, dataDir, ownerID string) string {
	volumeName := "openclaw-state-" + req.Container.ContainerName
	networkMode := req.Container.NetworkMode
	healthCommand := fmt.Sprintf(
		`node -e "fetch('http://127.0.0.1:%d/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"`,
		req.Container.GatewayPort,
	)
	if networkMode != "" && networkMode != "host" && !strings.HasPrefix(networkMode, "container:") {
		return generateOpenClawBridgeComposeYAML(req, dataDir, ownerID, volumeName, networkMode, healthCommand)
	}
	return generateOpenClawNetworkModeComposeYAML(req, dataDir, ownerID, volumeName, networkMode, healthCommand)
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
