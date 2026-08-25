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

type provisionFailure struct {
	message string
	status  int
}

type provisionWorkspace struct {
	containerDir         string
	workspaceDir         string
	configPath           string
	absoluteContainerDir string
}

type provisionOutcome struct {
	workspace   provisionWorkspace
	containerID string
}

func (h *OpenClawHandler) runSynchronousProvision(
	req *ProvisionRequest,
	requestedPortExplicit, bridgeMode bool,
) (provisionOutcome, *provisionFailure) {
	h.mu.Lock()
	outcome, failure := h.provisionLocked(req, requestedPortExplicit, bridgeMode)
	h.mu.Unlock()
	return outcome, failure
}

func (h *OpenClawHandler) provisionLocked(
	req *ProvisionRequest,
	requestedPortExplicit, bridgeMode bool,
) (provisionOutcome, *provisionFailure) {
	teams, teamName, failure := h.loadProvisionTeam(req.TeamID)
	if failure != nil {
		return provisionOutcome{}, failure
	}
	if failure := h.selectProvisionPort(req, bridgeMode); failure != nil {
		return provisionOutcome{}, failure
	}
	workspace, failure := h.prepareProvisionWorkspace(*req)
	if failure != nil {
		return provisionOutcome{}, failure
	}
	if err := h.ensureImageAvailable(req.Container.BaseImage); err != nil {
		return provisionOutcome{}, &provisionFailure{message: err.Error(), status: http.StatusBadRequest}
	}
	h.ensureProvisionNetwork(req.Container.NetworkMode)
	args := provisionContainerRunArgs(*req, workspace.absoluteContainerDir)
	containerID, failure := h.startProvisionContainer(req, workspace.configPath, args, requestedPortExplicit, bridgeMode)
	if failure != nil {
		return provisionOutcome{}, failure
	}
	h.persistProvisionRegistry(*req, teamName, workspace.absoluteContainerDir, teams)
	return provisionOutcome{workspace: workspace, containerID: containerID}, nil
}

func (h *OpenClawHandler) loadProvisionTeam(teamID string) ([]TeamEntry, string, *provisionFailure) {
	teams, err := h.loadTeams()
	if err != nil {
		return nil, "", &provisionFailure{
			message: fmt.Sprintf("Failed to load teams: %v", err), status: http.StatusInternalServerError,
		}
	}
	team := findTeamByID(teams, teamID)
	if team == nil {
		return nil, "", &provisionFailure{
			message: fmt.Sprintf("team %q not found", teamID), status: http.StatusNotFound,
		}
	}
	teamName := strings.TrimSpace(team.Name)
	if teamName == "" {
		return nil, "", &provisionFailure{
			message: fmt.Sprintf("team %q has empty name", teamID), status: http.StatusBadRequest,
		}
	}
	return teams, teamName, nil
}

func (h *OpenClawHandler) selectProvisionPort(req *ProvisionRequest, bridgeMode bool) *provisionFailure {
	if req.Container.GatewayPort == 0 {
		req.Container.GatewayPort = h.nextAvailablePort(req.Container.NetworkMode)
		return nil
	}
	if bridgeMode {
		return nil
	}
	entries, _ := h.loadRegistry()
	for _, entry := range entries {
		if entry.Port == req.Container.GatewayPort && entry.Name != req.Container.ContainerName {
			return &provisionFailure{
				message: fmt.Sprintf("Port %d already used by container %q", req.Container.GatewayPort, entry.Name),
				status:  http.StatusConflict,
			}
		}
	}
	if !isTCPPortAvailable(req.Container.GatewayPort) {
		return &provisionFailure{
			message: fmt.Sprintf(
				"Port %d is already in use on host. Stop the existing gateway/container (e.g. `openclaw gateway stop`) or choose another port.",
				req.Container.GatewayPort,
			),
			status: http.StatusConflict,
		}
	}
	return nil
}

func (h *OpenClawHandler) prepareProvisionWorkspace(req ProvisionRequest) (provisionWorkspace, *provisionFailure) {
	workspace := provisionWorkspace{containerDir: h.containerDataDir(req.Container.ContainerName)}
	workspace.workspaceDir = filepath.Join(workspace.containerDir, "workspace")
	workspace.configPath = filepath.Join(workspace.containerDir, "openclaw.json")
	for _, sub := range []string{"workspace", "workspace/memory", "workspace/skills"} {
		if err := os.MkdirAll(filepath.Join(workspace.containerDir, sub), 0o755); err != nil {
			return provisionWorkspace{}, &provisionFailure{
				message: fmt.Sprintf("Failed to create %s: %v", sub, err), status: http.StatusInternalServerError,
			}
		}
	}
	if err := writeIdentityFiles(workspace.workspaceDir, req.Identity); err != nil {
		return provisionWorkspace{}, &provisionFailure{
			message: fmt.Sprintf("Failed to write identity files: %v", err), status: http.StatusInternalServerError,
		}
	}
	if err := os.WriteFile(filepath.Join(workspace.workspaceDir, "AGENTS.md"), []byte(agentsMdContent()), 0o644); err != nil {
		return provisionWorkspace{}, &provisionFailure{
			message: fmt.Sprintf("Failed to write AGENTS.md: %v", err), status: http.StatusInternalServerError,
		}
	}
	h.installProvisionSkills(workspace.workspaceDir, req.Skills, req.Container.BaseImage)
	if err := writeOpenClawConfig(workspace.configPath, req); err != nil {
		return provisionWorkspace{}, &provisionFailure{
			message: fmt.Sprintf("Failed to write config: %v", err), status: http.StatusInternalServerError,
		}
	}
	workspace.absoluteContainerDir, _ = filepath.Abs(workspace.containerDir)
	return workspace, nil
}

func (h *OpenClawHandler) installProvisionSkills(workspaceDir string, skills []string, baseImage string) {
	for _, skillID := range skills {
		content := h.fetchSkillContent(skillID, baseImage)
		if content == "" {
			continue
		}
		skillDir := filepath.Join(workspaceDir, "skills", skillID)
		if err := os.MkdirAll(skillDir, 0o755); err != nil {
			log.Printf("openclaw: failed to create skill dir %s: %v", skillID, err)
			continue
		}
		if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte(content), 0o644); err != nil {
			log.Printf("openclaw: failed to write skill %s: %v", skillID, err)
		}
	}
}

func (h *OpenClawHandler) ensureProvisionNetwork(networkMode string) {
	if networkMode == "" || networkMode == "host" || strings.HasPrefix(networkMode, "container:") {
		return
	}
	_, createErr := h.containerCombinedOutput("network", "create", "--driver", "bridge", networkMode)
	if createErr == nil {
		return
	}
	if out, _ := h.containerCombinedOutput("network", "inspect", networkMode); len(out) == 0 {
		log.Printf("openclaw: warning: could not ensure network %s exists: %v", networkMode, createErr)
	}
}

func provisionContainerRunArgs(req ProvisionRequest, absoluteContainerDir string) []string {
	args := []string{
		"run", "-d", "--name", req.Container.ContainerName,
		"--user", "0:0", "--network", req.Container.NetworkMode,
	}
	healthCmd := fmt.Sprintf(
		"node -e \"fetch('http://127.0.0.1:%d/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))\"",
		req.Container.GatewayPort,
	)
	args = append(args,
		"--health-cmd", healthCmd,
		"--health-interval", "30s", "--health-timeout", "5s",
		"--health-start-period", "15s", "--health-retries", "3",
		"-v", absoluteContainerDir+"/workspace:/workspace",
		"-v", absoluteContainerDir+"/openclaw.json:/config/openclaw.json:ro",
		"-v", "openclaw-state-"+req.Container.ContainerName+":/state",
		"-e", "OPENCLAW_CONFIG_PATH=/config/openclaw.json",
		"-e", "OPENCLAW_STATE_DIR=/state",
		req.Container.BaseImage,
		"node", "openclaw.mjs", "gateway", "--allow-unconfigured", "--bind", "lan",
	)
	return args
}

func (h *OpenClawHandler) startProvisionContainer(
	req *ProvisionRequest,
	configPath string,
	args []string,
	requestedPortExplicit, bridgeMode bool,
) (string, *provisionFailure) {
	attemptLimit := 1
	if !bridgeMode && !requestedPortExplicit {
		attemptLimit = 4
	}
	for attempt := 0; attempt < attemptLimit; attempt++ {
		if failure := h.prepareProvisionRetry(req, configPath, attempt, attemptLimit); failure != nil {
			return "", failure
		}
		_ = h.containerRun("rm", "-f", req.Container.ContainerName)
		out, err := h.containerCombinedOutput(args...)
		if err != nil {
			retry, failure := provisionStartFailure(req, out, err, attempt, attemptLimit, requestedPortExplicit, bridgeMode)
			if retry {
				continue
			}
			return "", failure
		}
		containerID := strings.TrimSpace(string(out))
		if bridgeMode || requestedPortExplicit {
			return containerID, nil
		}
		conflictLogs := h.detectImmediateGatewayPortConflict(req.Container.ContainerName, req.Container.GatewayPort)
		if conflictLogs == "" {
			return containerID, nil
		}
		_ = h.containerRun("rm", "-f", req.Container.ContainerName)
		if attempt+1 >= attemptLimit {
			return "", &provisionFailure{
				message: fmt.Sprintf(
					"Gateway failed to bind port %d after %d attempts. Last error: %s",
					req.Container.GatewayPort, attemptLimit, truncatePortConflictLog(conflictLogs),
				),
				status: http.StatusConflict,
			}
		}
		log.Printf("openclaw: detected gateway port conflict for %q on %d; retrying with a new port", req.Container.ContainerName, req.Container.GatewayPort)
	}
	return "", &provisionFailure{message: "Failed to start container", status: http.StatusInternalServerError}
}

func (h *OpenClawHandler) prepareProvisionRetry(
	req *ProvisionRequest,
	configPath string,
	attempt, attemptLimit int,
) *provisionFailure {
	if attempt == 0 {
		return nil
	}
	req.Container.GatewayPort = h.nextAvailablePort(req.Container.NetworkMode)
	if err := writeOpenClawConfig(configPath, *req); err != nil {
		return &provisionFailure{
			message: fmt.Sprintf("Failed to refresh config for port retry: %v", err),
			status:  http.StatusInternalServerError,
		}
	}
	log.Printf(
		"openclaw: retrying %q with alternate port %d (attempt %d/%d)",
		req.Container.ContainerName, req.Container.GatewayPort, attempt+1, attemptLimit,
	)
	return nil
}

func provisionStartFailure(
	req *ProvisionRequest,
	out []byte,
	err error,
	attempt, attemptLimit int,
	requestedPortExplicit, bridgeMode bool,
) (bool, *provisionFailure) {
	trimmed := strings.TrimSpace(string(out))
	if isContainerImageMissingError(trimmed) {
		return false, &provisionFailure{
			message: fmt.Sprintf(
				"OpenClaw image %q is unavailable on host runtime. Build or pull this image first, or set OPENCLAW_BASE_IMAGE to an available image before starting dashboard.",
				req.Container.BaseImage,
			),
			status: http.StatusBadRequest,
		}
	}
	if !bridgeMode && !requestedPortExplicit && isOpenClawGatewayPortConflict(trimmed, req.Container.GatewayPort) && attempt+1 < attemptLimit {
		log.Printf("openclaw: container runtime start failed due to port conflict on %d, retrying: %s", req.Container.GatewayPort, trimmed)
		return true, nil
	}
	return false, &provisionFailure{
		message: fmt.Sprintf("Failed to start container: %s (%v)", trimmed, err),
		status:  http.StatusInternalServerError,
	}
}

func (h *OpenClawHandler) persistProvisionRegistry(
	req ProvisionRequest,
	teamName, absoluteContainerDir string,
	teams []TeamEntry,
) {
	entries, _ := h.loadRegistry()
	found := false
	for index := range entries {
		if entries[index].Name == req.Container.ContainerName {
			updateProvisionRegistryEntry(&entries[index], req, teamName, absoluteContainerDir)
			found = true
			break
		}
	}
	if !found {
		entry := ContainerEntry{CreatedAt: time.Now().UTC().Format(time.RFC3339)}
		updateProvisionRegistryEntry(&entry, req, teamName, absoluteContainerDir)
		entries = append(entries, entry)
	}
	if req.RoleKind == "leader" {
		promoteProvisionLeader(entries, teams, req)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
	if err := h.saveRegistry(entries); err != nil {
		log.Printf("openclaw: failed to save registry: %v", err)
	}
	if err := h.saveTeams(teams); err != nil {
		log.Printf("openclaw: failed to save teams after provisioning: %v", err)
	}
}

func updateProvisionRegistryEntry(entry *ContainerEntry, req ProvisionRequest, teamName, dataDir string) {
	entry.Name = req.Container.ContainerName
	entry.Port = req.Container.GatewayPort
	entry.Image = req.Container.BaseImage
	entry.Token = req.Container.AuthToken
	entry.DataDir = dataDir
	entry.TeamID = req.TeamID
	entry.TeamName = teamName
	entry.AgentName = strings.TrimSpace(req.Identity.Name)
	entry.AgentEmoji = strings.TrimSpace(req.Identity.Emoji)
	entry.AgentRole = strings.TrimSpace(req.Identity.Role)
	entry.AgentVibe = strings.TrimSpace(req.Identity.Vibe)
	entry.AgentPrinciples = strings.TrimSpace(req.Identity.Principles)
	entry.RoleKind = req.RoleKind
}

func promoteProvisionLeader(entries []ContainerEntry, teams []TeamEntry, req ProvisionRequest) {
	for index := range entries {
		if entries[index].TeamID == req.TeamID && entries[index].Name != req.Container.ContainerName {
			entries[index].RoleKind = "worker"
		}
	}
	for index := range teams {
		if teams[index].ID == req.TeamID {
			teams[index].LeaderID = req.Container.ContainerName
			teams[index].UpdatedAt = time.Now().UTC().Format(time.RFC3339)
			return
		}
	}
}

func (h *OpenClawHandler) writeProvisionOutcome(
	w http.ResponseWriter,
	req ProvisionRequest,
	runtimeName string,
	outcome provisionOutcome,
) {
	healthy := false
	for attempt := 0; attempt < 10; attempt++ {
		time.Sleep(2 * time.Second)
		if h.gatewayHealthyForContainer(req.Container.ContainerName, req.Container.GatewayPort) {
			healthy = true
			break
		}
	}
	message := "Container started and gateway is healthy"
	if !healthy {
		message = "Container started but gateway has not become healthy yet (may still be initializing)"
	}
	log.Printf("OpenClaw provisioned: name=%s port=%d healthy=%v", req.Container.ContainerName, req.Container.GatewayPort, healthy)
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(ProvisionResponse{
		Success: true, Message: message,
		WorkspaceDir: outcome.workspace.workspaceDir, ConfigPath: outcome.workspace.configPath,
		ContainerID: outcome.containerID,
		DockerCmd:   generateDockerRunCmd(runtimeName, req, outcome.workspace.absoluteContainerDir),
		ComposeYAML: generateComposeYAML(req, outcome.workspace.absoluteContainerDir),
	}); err != nil {
		log.Printf("openclaw: provision encode error: %v", err)
	}
}
