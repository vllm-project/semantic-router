package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

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
		req.TeamID = sanitizeTeamID(req.TeamID)
		if req.TeamID == "" {
			writeJSONError(w, "teamId is required; create/select a team before provisioning", http.StatusBadRequest)
			return
		}

		h.mu.Lock()
		teams, err := h.loadTeams()
		if err != nil {
			h.mu.Unlock()
			writeJSONError(w, fmt.Sprintf("Failed to load teams: %v", err), http.StatusInternalServerError)
			return
		}
		team := findTeamByID(teams, req.TeamID)
		if team == nil {
			h.mu.Unlock()
			writeJSONError(w, fmt.Sprintf("team %q not found", req.TeamID), http.StatusNotFound)
			return
		}
		teamName := strings.TrimSpace(team.Name)
		if teamName == "" {
			h.mu.Unlock()
			writeJSONError(w, fmt.Sprintf("team %q has empty name", req.TeamID), http.StatusBadRequest)
			return
		}

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
				entries[i].TeamID = req.TeamID
				entries[i].TeamName = teamName
				entries[i].AgentName = strings.TrimSpace(req.Identity.Name)
				entries[i].AgentEmoji = strings.TrimSpace(req.Identity.Emoji)
				entries[i].AgentRole = strings.TrimSpace(req.Identity.Role)
				entries[i].AgentVibe = strings.TrimSpace(req.Identity.Vibe)
				entries[i].AgentPrinciples = strings.TrimSpace(req.Identity.Principles)
				found = true
				break
			}
		}
		if !found {
			entries = append(entries, ContainerEntry{
				Name:            req.Container.ContainerName,
				Port:            req.Container.GatewayPort,
				Image:           req.Container.BaseImage,
				Token:           req.Container.AuthToken,
				DataDir:         absCDir,
				CreatedAt:       time.Now().UTC().Format(time.RFC3339),
				TeamID:          req.TeamID,
				TeamName:        teamName,
				AgentName:       strings.TrimSpace(req.Identity.Name),
				AgentEmoji:      strings.TrimSpace(req.Identity.Emoji),
				AgentRole:       strings.TrimSpace(req.Identity.Role),
				AgentVibe:       strings.TrimSpace(req.Identity.Vibe),
				AgentPrinciples: strings.TrimSpace(req.Identity.Principles),
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
