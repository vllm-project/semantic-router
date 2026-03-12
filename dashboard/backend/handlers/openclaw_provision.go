package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
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
		if !h.canManageOpenClaw() {
			h.writeReadOnlyError(w)
			return
		}

		asyncRequested := provisionAsyncRequested(r)

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
			if reusedToken := h.gatewayTokenForContainer(req.Container.ContainerName); reusedToken != "" {
				req.Container.AuthToken = reusedToken
			} else {
				req.Container.AuthToken = generateToken(24)
			}
		}
		req.Container.BaseImage = h.resolveBaseImage(req.Container.BaseImage)
		if preferredNetwork := strings.TrimSpace(os.Getenv("OPENCLAW_DEFAULT_NETWORK_MODE")); preferredNetwork != "" {
			// In vllm-sr serve deployment, dashboard often runs in a container while OpenClaw
			// is launched via host docker.sock. Using container:<dashboard-container> keeps
			// gateway traffic in the same network namespace and avoids host routing issues.
			// Override generic values ("", "host", "bridge") with the preferred network so
			// that the container is placed on the same user-defined bridge network as the
			// dashboard (Docker's default "bridge" network does not support container-name
			// DNS resolution).
			nm := strings.ToLower(strings.TrimSpace(req.Container.NetworkMode))
			if nm == "" || nm == "host" || nm == "bridge" {
				req.Container.NetworkMode = preferredNetwork
			}
		}
		if req.Container.NetworkMode == "" {
			req.Container.NetworkMode = "host"
		}
		// When using a user-defined bridge network, the OpenClaw container
		// reaches the SR router via container-name DNS, not localhost.
		// Automatically rewrite loopback addresses in modelBaseUrl to the
		// dashboard container name so users don't have to do it manually.
		if nm := req.Container.NetworkMode; nm != "host" && !strings.HasPrefix(nm, "container:") {
			dashboardContainer := strings.TrimSpace(os.Getenv("OPENCLAW_DASHBOARD_CONTAINER_NAME"))
			if dashboardContainer == "" {
				dashboardContainer = vllmSrContainerName
			}
			if req.Container.ModelBaseURL == "" {
				req.Container.ModelBaseURL = h.resolveOpenClawModelBaseURL()
			}
			req.Container.ModelBaseURL = rewriteLoopbackHost(req.Container.ModelBaseURL, dashboardContainer)
			if req.Container.MemoryBaseURL != "" {
				req.Container.MemoryBaseURL = rewriteLoopbackHost(req.Container.MemoryBaseURL, dashboardContainer)
			}
		} else if req.Container.ModelBaseURL == "" {
			req.Container.ModelBaseURL = h.resolveOpenClawModelBaseURL()
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
		req.RoleKind = normalizeRoleKind(req.RoleKind)
		requestedPortExplicit := req.Container.GatewayPort != 0
		bridgeMode := isBridgeNetwork(req.Container.NetworkMode)

		if asyncRequested {
			reqCopy := req
			go h.runProvisionAsync(reqCopy)

			log.Printf("OpenClaw provision queued async: name=%s team=%s", req.Container.ContainerName, req.TeamID)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)
			if err := json.NewEncoder(w).Encode(ProvisionResponse{
				Success:      true,
				Message:      "Provision request accepted; worker creation is running asynchronously",
				WorkspaceDir: filepath.Join(h.containerDataDir(req.Container.ContainerName), "workspace"),
				ConfigPath:   filepath.Join(h.containerDataDir(req.Container.ContainerName), "openclaw.json"),
			}); err != nil {
				log.Printf("openclaw: provision encode error: %v", err)
			}
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
			req.Container.GatewayPort = h.nextAvailablePort(req.Container.NetworkMode)
		} else if !bridgeMode {
			// In host network mode, check for port conflicts
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
		// In bridge mode with explicit port: no host-level conflict check needed
		// because each container has its own network namespace

		cDir := h.containerDataDir(req.Container.ContainerName)
		wsDir := filepath.Join(cDir, "workspace")
		for _, sub := range []string{
			"workspace",
			"workspace/memory",
			"workspace/skills",
		} {
			err = os.MkdirAll(filepath.Join(cDir, sub), 0o755)
			if err != nil {
				h.mu.Unlock()
				writeJSONError(w, fmt.Sprintf("Failed to create %s: %v", sub, err), http.StatusInternalServerError)
				return
			}
		}

		err = writeIdentityFiles(wsDir, req.Identity)
		if err != nil {
			h.mu.Unlock()
			writeJSONError(w, fmt.Sprintf("Failed to write identity files: %v", err), http.StatusInternalServerError)
			return
		}
		err = os.WriteFile(filepath.Join(wsDir, "AGENTS.md"), []byte(agentsMdContent()), 0o644)
		if err != nil {
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
			err = os.MkdirAll(skillDir, 0o755)
			if err != nil {
				log.Printf("openclaw: failed to create skill dir %s: %v", skillID, err)
				continue
			}
			err = os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte(content), 0o644)
			if err != nil {
				log.Printf("openclaw: failed to write skill %s: %v", skillID, err)
			}
		}

		configPath := filepath.Join(cDir, "openclaw.json")
		err = writeOpenClawConfig(configPath, req)
		if err != nil {
			h.mu.Unlock()
			writeJSONError(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
			return
		}

		err = h.ensureImageAvailable(req.Container.BaseImage)
		if err != nil {
			h.mu.Unlock()
			writeJSONError(w, err.Error(), http.StatusBadRequest)
			return
		}

		// For bridge network names, ensure the network exists before starting
		// the container. This is idempotent: if the network already exists the
		// command exits silently.
		networkMode := req.Container.NetworkMode
		if networkMode != "" && networkMode != "host" && !strings.HasPrefix(networkMode, "container:") {
			if _, err := h.containerCombinedOutput("network", "create", "--driver", "bridge", networkMode); err != nil {
				// "already exists" is expected and harmless.
				if out, _ := h.containerCombinedOutput("network", "inspect", networkMode); len(out) == 0 {
					log.Printf("openclaw: warning: could not ensure network %s exists: %v", networkMode, err)
				}
			}
		}

		absCDir, _ := filepath.Abs(cDir)
		volumeName := "openclaw-state-" + req.Container.ContainerName
		args := []string{
			"run", "-d",
			"--name", req.Container.ContainerName,
			"--user", "0:0",
			"--network", networkMode,
		}
		// Override the image's built-in healthcheck to point at the actual gateway port.
		healthCmd := fmt.Sprintf(
			"node -e \"fetch('http://127.0.0.1:%d/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))\"",
			req.Container.GatewayPort,
		)
		args = append(args,
			"--health-cmd", healthCmd,
			"--health-interval", "30s",
			"--health-timeout", "5s",
			"--health-start-period", "15s",
			"--health-retries", "3",
		)
		args = append(args,
			"-v", absCDir+"/workspace:/workspace",
			"-v", absCDir+"/openclaw.json:/config/openclaw.json:ro",
			"-v", volumeName+":/state",
			"-e", "OPENCLAW_CONFIG_PATH=/config/openclaw.json",
			"-e", "OPENCLAW_STATE_DIR=/state",
			req.Container.BaseImage,
			"node", "openclaw.mjs", "gateway", "--allow-unconfigured", "--bind", "lan",
		)
		// In bridge mode, no port conflict retry needed since containers have isolated namespaces.
		// In host mode, retry with alternate ports if user didn't explicitly request a port.
		startAttemptLimit := 1
		if !bridgeMode && !requestedPortExplicit {
			startAttemptLimit = 4
		}

		var containerID string
		for attempt := 0; attempt < startAttemptLimit; attempt++ {
			if attempt > 0 {
				req.Container.GatewayPort = h.nextAvailablePort(req.Container.NetworkMode)
				if err := writeOpenClawConfig(configPath, req); err != nil {
					h.mu.Unlock()
					writeJSONError(w, fmt.Sprintf("Failed to refresh config for port retry: %v", err), http.StatusInternalServerError)
					return
				}
				log.Printf(
					"openclaw: retrying %q with alternate port %d (attempt %d/%d)",
					req.Container.ContainerName,
					req.Container.GatewayPort,
					attempt+1,
					startAttemptLimit,
				)
			}

			_ = h.containerRun("rm", "-f", req.Container.ContainerName)

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
				// Only retry port conflicts in host mode
				if !bridgeMode && !requestedPortExplicit && isOpenClawGatewayPortConflict(trimmed, req.Container.GatewayPort) && attempt+1 < startAttemptLimit {
					log.Printf(
						"openclaw: container runtime start failed due to port conflict on %d, retrying: %s",
						req.Container.GatewayPort,
						trimmed,
					)
					continue
				}

				h.mu.Unlock()
				writeJSONError(w, fmt.Sprintf("Failed to start container: %s (%v)", trimmed, err), http.StatusInternalServerError)
				return
			}

			containerID = strings.TrimSpace(string(out))
			if bridgeMode || requestedPortExplicit {
				break
			}

			conflictLogs := h.detectImmediateGatewayPortConflict(req.Container.ContainerName, req.Container.GatewayPort)
			if conflictLogs == "" {
				break
			}

			_ = h.containerRun("rm", "-f", req.Container.ContainerName)
			if attempt+1 >= startAttemptLimit {
				h.mu.Unlock()
				writeJSONError(
					w,
					fmt.Sprintf(
						"Gateway failed to bind port %d after %d attempts. Last error: %s",
						req.Container.GatewayPort,
						startAttemptLimit,
						truncatePortConflictLog(conflictLogs),
					),
					http.StatusConflict,
				)
				return
			}
			log.Printf(
				"openclaw: detected gateway port conflict for %q on %d; retrying with a new port",
				req.Container.ContainerName,
				req.Container.GatewayPort,
			)
		}

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
				entries[i].RoleKind = req.RoleKind
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
				RoleKind:        req.RoleKind,
			})
		}
		if req.RoleKind == "leader" {
			for i := range entries {
				if entries[i].TeamID == req.TeamID && entries[i].Name != req.Container.ContainerName {
					entries[i].RoleKind = "worker"
				}
			}
			for i := range teams {
				if teams[i].ID == req.TeamID {
					teams[i].LeaderID = req.Container.ContainerName
					teams[i].UpdatedAt = time.Now().UTC().Format(time.RFC3339)
					break
				}
			}
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
		if err := h.saveRegistry(entries); err != nil {
			log.Printf("openclaw: failed to save registry: %v", err)
		}
		if err := h.saveTeams(teams); err != nil {
			log.Printf("openclaw: failed to save teams after provisioning: %v", err)
		}
		h.mu.Unlock()

		dockerCmd := generateDockerRunCmd(runtimeName, req, absCDir)
		composeYAML := generateComposeYAML(req, absCDir)

		healthy := false
		for i := 0; i < 10; i++ {
			time.Sleep(2 * time.Second)
			if h.gatewayHealthyForContainer(req.Container.ContainerName, req.Container.GatewayPort) {
				healthy = true
				break
			}
		}

		msg := "Container started and gateway is healthy"
		if !healthy {
			msg = "Container started but gateway has not become healthy yet (may still be initializing)"
		}

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

func provisionAsyncRequested(r *http.Request) bool {
	if r == nil {
		return false
	}

	parseBool := func(raw string) bool {
		switch strings.ToLower(strings.TrimSpace(raw)) {
		case "1", "true", "yes", "on":
			return true
		default:
			return false
		}
	}

	if parseBool(r.URL.Query().Get("async")) {
		return true
	}
	return parseBool(r.Header.Get("X-OpenClaw-Async"))
}

func (h *OpenClawHandler) runProvisionAsync(req ProvisionRequest) {
	raw, err := json.Marshal(req)
	if err != nil {
		log.Printf("openclaw: async provision marshal failed for %s: %v", req.Container.ContainerName, err)
		return
	}

	internalReq := httptest.NewRequest(http.MethodPost, "/api/openclaw/workers", bytes.NewReader(raw))
	internalReq.Header.Set("Content-Type", "application/json")

	recorder := httptest.NewRecorder()
	h.ProvisionHandler().ServeHTTP(recorder, internalReq)

	if recorder.Code >= http.StatusOK && recorder.Code < http.StatusMultipleChoices {
		log.Printf("openclaw: async provision completed for %s (status=%d)", req.Container.ContainerName, recorder.Code)
		return
	}

	log.Printf(
		"openclaw: async provision failed for %s (status=%d): %s",
		req.Container.ContainerName,
		recorder.Code,
		strings.TrimSpace(recorder.Body.String()),
	)
}

func isOpenClawGatewayPortConflict(output string, port int) bool {
	text := strings.ToLower(strings.TrimSpace(output))
	if text == "" {
		return false
	}
	if strings.Contains(text, "another gateway instance is already listening") {
		return true
	}
	if strings.Contains(text, "address already in use") {
		return true
	}
	if strings.Contains(text, "port") && strings.Contains(text, "already in use") {
		return true
	}
	if port > 0 {
		portToken := fmt.Sprintf(":%d", port)
		if strings.Contains(text, portToken) && strings.Contains(text, "in use") {
			return true
		}
	}
	return false
}

func (h *OpenClawHandler) detectImmediateGatewayPortConflict(containerName string, port int) string {
	for i := 0; i < 80; i++ {
		time.Sleep(500 * time.Millisecond)
		logsOut, err := h.containerCombinedOutput("logs", "--tail", "120", containerName)
		if err != nil {
			continue
		}
		logs := strings.TrimSpace(string(logsOut))
		if isOpenClawGatewayPortConflict(logs, port) {
			return logs
		}
		if openClawGatewayListeningReady(logs) && h.gatewayReachable(containerName, port) {
			return ""
		}
	}
	return ""
}

func (h *OpenClawHandler) gatewayHealthyForContainer(containerName string, port int) bool {
	if !h.containerRunning(containerName) {
		return false
	}

	logsOut, err := h.containerCombinedOutput("logs", "--tail", "120", containerName)
	if err != nil {
		return false
	}
	logs := strings.TrimSpace(string(logsOut))
	if isOpenClawGatewayPortConflict(logs, port) {
		return false
	}
	if !openClawGatewayListeningReady(logs) {
		return false
	}
	return h.gatewayReachable(containerName, port)
}

func (h *OpenClawHandler) containerRunning(containerName string) bool {
	out, err := h.containerOutput("inspect", "-f", "{{.State.Running}}", containerName)
	if err != nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(string(out)), "true")
}

func openClawGatewayListeningReady(logs string) bool {
	text := strings.TrimSpace(logs)
	if text == "" {
		return false
	}

	lastSuccess := strings.LastIndex(text, "[gateway] listening on ws://")
	if lastSuccess < 0 {
		return false
	}

	lastFail := max(
		strings.LastIndex(text, "failed to start:"),
		strings.LastIndex(text, "permission denied, mkdir '/state/"),
		strings.LastIndex(text, "another gateway instance is already listening"),
		strings.LastIndex(text, "already in use"),
	)

	return lastSuccess > lastFail
}

func truncatePortConflictLog(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if len(trimmed) <= 280 {
		return trimmed
	}
	return strings.TrimSpace(trimmed[:277]) + "..."
}
