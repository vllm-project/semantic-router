package handlers

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

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

	identityContent, err := os.ReadFile(filepath.Join(wsDir, "IDENTITY.md"))
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

	soulContent, err := os.ReadFile(filepath.Join(wsDir, "SOUL.md"))
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
			"http": map[string]interface{}{
				"endpoints": map[string]interface{}{
					"chatCompletions": map[string]interface{}{"enabled": true},
					"responses":       map[string]interface{}{"enabled": true},
				},
			},
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
	healthCmd := fmt.Sprintf(
		`node -e "fetch('http://127.0.0.1:%d/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"`,
		req.Container.GatewayPort,
	)
	return fmt.Sprintf(`%s run -d \
  --name %s \
  --user 0:0 \
  --network %s \
  --health-cmd '%s' \
  --health-interval 30s \
  --health-timeout 5s \
  --health-start-period 15s \
  --health-retries 3 \
  -v %s/workspace:/workspace \
  -v %s/openclaw.json:/config/openclaw.json:ro \
  -v %s:/state \
  -e OPENCLAW_CONFIG_PATH=/config/openclaw.json \
  -e OPENCLAW_STATE_DIR=/state \
  %s \
  node openclaw.mjs gateway --allow-unconfigured --bind lan`,
		runtime, req.Container.ContainerName, req.Container.NetworkMode, healthCmd,
		dataDir, dataDir, volumeName, req.Container.BaseImage)
}

func generateComposeYAML(req ProvisionRequest, dataDir string) string {
	volumeName := "openclaw-state-" + req.Container.ContainerName
	networkMode := req.Container.NetworkMode

	// For bridge network names (not "host" or "container:xxx"), use the networks syntax.
	if networkMode != "" && networkMode != "host" && !strings.HasPrefix(networkMode, "container:") {
		return fmt.Sprintf(`services:
  openclaw:
    image: %s
    container_name: %s
    user: "0:0"
    networks:
      - %s
    volumes:
      - %s/workspace:/workspace
      - %s/openclaw.json:/config/openclaw.json:ro
      - %s:/state
    environment:
      OPENCLAW_CONFIG_PATH: /config/openclaw.json
      OPENCLAW_STATE_DIR: /state
    healthcheck:
      test: ["CMD-SHELL", "node -e \"fetch('http://127.0.0.1:%d/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))\""]
      interval: 30s
      timeout: 5s
      start_period: 15s
      retries: 3
    command: ["node", "openclaw.mjs", "gateway", "--allow-unconfigured", "--bind", "lan"]
    restart: unless-stopped

networks:
  %s:
    external: true

volumes:
  %s:
`, req.Container.BaseImage, req.Container.ContainerName, networkMode,
			dataDir, dataDir, volumeName,
			req.Container.GatewayPort,
			networkMode, volumeName)
	}

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
    healthcheck:
      test: ["CMD-SHELL", "node -e \"fetch('http://127.0.0.1:%d/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))\""]
      interval: 30s
      timeout: 5s
      start_period: 15s
      retries: 3
    command: ["node", "openclaw.mjs", "gateway", "--allow-unconfigured", "--bind", "lan"]
    restart: unless-stopped

volumes:
  %s:
`, req.Container.BaseImage, req.Container.ContainerName, networkMode,
		dataDir, dataDir, volumeName,
		req.Container.GatewayPort,
		volumeName)
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
