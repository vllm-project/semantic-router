package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type openClawProvisionFailure struct {
	message string
	status  int
}

type openClawProvisionPlan struct {
	request               ProvisionRequest
	runtimeName           string
	productionSecurity    bool
	requestedPortExplicit bool
	bridgeMode            bool
}

type openClawProvisionState struct {
	teams         []TeamEntry
	originalTeams []TeamEntry
	entries       []ContainerEntry
	teamName      string
}

type openClawProvisionArtifacts struct {
	containerDir string
	workspaceDir string
	configPath   string
	ownerID      string
}

type openClawProvisionResult struct {
	plan        *openClawProvisionPlan
	artifacts   openClawProvisionArtifacts
	containerID string
}

func (h *OpenClawHandler) handleProvisionRequest(w http.ResponseWriter, r *http.Request) {
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
		writeJSONError(w, "OpenClaw container runtime is unavailable", http.StatusServiceUnavailable)
		return
	}
	runtimeName := filepath.Base(runtimeBin)
	if runtimeName == "" {
		runtimeName = runtimeBin
	}

	var req ProvisionRequest
	if status, err := decodeBoundedJSON(w, r, maximumOpenClawProvisionRequestBytes, &req); err != nil {
		writeJSONError(w, err.Error(), status)
		return
	}
	if err := validateOpenClawProvisionRequest(req); err != nil {
		writeJSONError(w, err.Error(), http.StatusBadRequest)
		return
	}
	plan, failure := h.prepareOpenClawProvisionPlan(req, runtimeName)
	if failure != nil {
		writeJSONError(w, failure.message, failure.status)
		return
	}
	if asyncRequested {
		if plan.productionSecurity {
			writeJSONError(w, "asynchronous OpenClaw provisioning is disabled in production", http.StatusBadRequest)
			return
		}
		h.queueOpenClawProvision(w, plan)
		return
	}
	if failure = h.prepareOpenClawProvisionSecurity(plan); failure != nil {
		writeJSONError(w, failure.message, failure.status)
		return
	}

	h.mu.Lock()
	result, failure := h.provisionOpenClawLocked(plan)
	h.mu.Unlock()
	if failure != nil {
		writeJSONError(w, failure.message, failure.status)
		return
	}
	h.writeOpenClawProvisionSuccess(w, result)
}

func (h *OpenClawHandler) prepareOpenClawProvisionPlan(
	req ProvisionRequest,
	runtimeName string,
) (*openClawProvisionPlan, *openClawProvisionFailure) {
	req.Container.ContainerName = deriveContainerName(req.Container.ContainerName, req.Identity.Name)
	productionSecurity := dashboardUsesProductionSecurityProfile()
	req.Container.BaseImage = h.resolveBaseImage(req.Container.BaseImage)
	resolvedNetwork, err := resolveOpenClawProvisionNetworkMode(
		req.Container.NetworkMode,
		os.Getenv("OPENCLAW_DEFAULT_NETWORK_MODE"),
		productionSecurity,
	)
	if err != nil {
		if errors.Is(err, errOpenClawProductionNetworkConfiguration) {
			return nil, &openClawProvisionFailure{
				message: "Production OpenClaw network is not configured", status: http.StatusServiceUnavailable,
			}
		}
		return nil, &openClawProvisionFailure{
			message: "networkMode must be omitted, generic host/bridge, or match the configured production network",
			status:  http.StatusBadRequest,
		}
	}
	req.Container.NetworkMode = resolvedNetwork
	if err := validateOpenClawImageReference(req.Container.BaseImage, productionSecurity); err != nil {
		return nil, &openClawProvisionFailure{message: err.Error(), status: http.StatusBadRequest}
	}
	if err := validateOpenClawNetworkMode(req.Container.NetworkMode, productionSecurity); err != nil {
		return nil, &openClawProvisionFailure{message: err.Error(), status: http.StatusBadRequest}
	}
	if productionSecurity && req.Container.BrowserEnabled {
		return nil, &openClawProvisionFailure{
			message: "production OpenClaw workers cannot disable the browser sandbox", status: http.StatusBadRequest,
		}
	}
	h.applyOpenClawProvisionModelDefaults(&req)
	req.TeamID = sanitizeTeamID(req.TeamID)
	if req.TeamID == "" {
		return nil, &openClawProvisionFailure{
			message: "teamId is required; create/select a team before provisioning", status: http.StatusBadRequest,
		}
	}
	validatedSkills, skillsErr := validateRequestedOpenClawSkills(req.Skills)
	if skillsErr != nil {
		return nil, &openClawProvisionFailure{message: skillsErr.Error(), status: http.StatusBadRequest}
	}
	req.Skills = validatedSkills
	req.RoleKind = normalizeRoleKind(req.RoleKind)
	return &openClawProvisionPlan{
		request:               req,
		runtimeName:           runtimeName,
		productionSecurity:    productionSecurity,
		requestedPortExplicit: req.Container.GatewayPort != 0,
		bridgeMode:            isBridgeNetwork(req.Container.NetworkMode),
	}, nil
}

func (h *OpenClawHandler) applyOpenClawProvisionModelDefaults(req *ProvisionRequest) {
	networkMode := req.Container.NetworkMode
	if networkMode != "host" && !strings.HasPrefix(networkMode, "container:") {
		modelGatewayContainer := openClawModelGatewayContainerName()
		if req.Container.ModelBaseURL == "" {
			req.Container.ModelBaseURL = h.resolveOpenClawModelBaseURL()
		}
		req.Container.ModelBaseURL = rewriteLoopbackHost(req.Container.ModelBaseURL, modelGatewayContainer)
		if req.Container.MemoryBaseURL != "" {
			req.Container.MemoryBaseURL = rewriteLoopbackHost(req.Container.MemoryBaseURL, modelGatewayContainer)
		}
	} else if req.Container.ModelBaseURL == "" {
		req.Container.ModelBaseURL = h.resolveOpenClawModelBaseURL()
	}
	if req.Container.ModelAPIKey == "" {
		req.Container.ModelAPIKey = "not-needed"
	}
	if req.Container.ModelName == "" {
		req.Container.ModelName = "vllm-sr/auto"
	}
	if req.Container.MemoryBackend == "" {
		req.Container.MemoryBackend = "local"
	}
}

func (h *OpenClawHandler) queueOpenClawProvision(w http.ResponseWriter, plan *openClawProvisionPlan) {
	reqCopy := plan.request
	go h.runProvisionAsync(reqCopy)
	log.Printf("OpenClaw provision queued async: name=%s team=%s", plan.request.Container.ContainerName, plan.request.TeamID)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	if err := json.NewEncoder(w).Encode(publicOpenClawProvisionResponse(false, ProvisionResponse{
		Success:      true,
		Message:      "Provision request accepted; worker creation is running asynchronously",
		WorkspaceDir: filepath.Join(h.containerDataDir(plan.request.Container.ContainerName), "workspace"),
		ConfigPath:   filepath.Join(h.containerDataDir(plan.request.Container.ContainerName), "openclaw.json"),
	})); err != nil {
		log.Printf("openclaw: provision encode error: %v", err)
	}
}

func (h *OpenClawHandler) prepareOpenClawProvisionSecurity(plan *openClawProvisionPlan) *openClawProvisionFailure {
	if plan.productionSecurity {
		if err := h.ensureOpenClawProvisionNetwork(plan.request.Container.NetworkMode, true); err != nil {
			return &openClawProvisionFailure{
				message: "Configured production OpenClaw network is unavailable", status: http.StatusServiceUnavailable,
			}
		}
	}
	resolvedToken, err := h.resolveOpenClawProvisionToken(
		plan.request.Container.ContainerName,
		plan.request.Container.AuthToken,
		plan.productionSecurity,
	)
	if err != nil {
		return &openClawProvisionFailure{
			message: "Failed to generate worker credential", status: http.StatusInternalServerError,
		}
	}
	plan.request.Container.AuthToken = resolvedToken
	return nil
}

func (h *OpenClawHandler) provisionOpenClawLocked(
	plan *openClawProvisionPlan,
) (*openClawProvisionResult, *openClawProvisionFailure) {
	state, failure := h.loadOpenClawProvisionState(plan)
	if failure != nil {
		return nil, failure
	}
	artifacts, failure := h.prepareOpenClawProvisionArtifacts(plan)
	if failure != nil {
		return nil, failure
	}
	containerID, failure := h.startOpenClawProvisionContainer(plan, artifacts)
	if failure != nil {
		return nil, failure
	}
	if failure := h.persistOpenClawProvisionState(plan, state, artifacts, containerID); failure != nil {
		return nil, failure
	}
	return &openClawProvisionResult{plan: plan, artifacts: artifacts, containerID: containerID}, nil
}

func (h *OpenClawHandler) loadOpenClawProvisionState(
	plan *openClawProvisionPlan,
) (*openClawProvisionState, *openClawProvisionFailure) {
	teams, err := h.loadTeams()
	if err != nil {
		return nil, &openClawProvisionFailure{message: "Failed to load OpenClaw teams", status: http.StatusInternalServerError}
	}
	team := findTeamByID(teams, plan.request.TeamID)
	if team == nil {
		return nil, &openClawProvisionFailure{
			message: fmt.Sprintf("team %q not found", plan.request.TeamID), status: http.StatusNotFound,
		}
	}
	teamName := strings.TrimSpace(team.Name)
	if teamName == "" {
		return nil, &openClawProvisionFailure{
			message: fmt.Sprintf("team %q has empty name", plan.request.TeamID), status: http.StatusBadRequest,
		}
	}
	entries, err := h.loadRegistry()
	if err != nil {
		return nil, &openClawProvisionFailure{message: "Failed to load OpenClaw workers", status: http.StatusInternalServerError}
	}
	if !plan.productionSecurity {
		if err := h.ensureOpenClawProvisionNetwork(plan.request.Container.NetworkMode, false); err != nil {
			return nil, &openClawProvisionFailure{message: "Failed to ensure OpenClaw network", status: http.StatusInternalServerError}
		}
	}
	if plan.request.Container.GatewayPort == 0 {
		for _, entry := range entries {
			if entry.Name == plan.request.Container.ContainerName && entry.Port > 0 {
				plan.request.Container.GatewayPort = entry.Port
				break
			}
		}
	}
	if err := h.ensureImageAvailable(plan.request.Container.BaseImage); err != nil {
		return nil, &openClawProvisionFailure{message: err.Error(), status: http.StatusBadRequest}
	}
	if err := h.removeOwnedContainerIfPresent(plan.request.Container.ContainerName); err != nil {
		return nil, &openClawProvisionFailure{message: openClawLifecyclePublicError(err), status: openClawLifecycleHTTPStatus(err)}
	}
	if failure := h.selectOpenClawProvisionPort(plan, entries); failure != nil {
		return nil, failure
	}
	return &openClawProvisionState{
		teams: teams, originalTeams: append([]TeamEntry(nil), teams...), entries: entries, teamName: teamName,
	}, nil
}

func (h *OpenClawHandler) selectOpenClawProvisionPort(
	plan *openClawProvisionPlan,
	entries []ContainerEntry,
) *openClawProvisionFailure {
	if plan.request.Container.GatewayPort == 0 {
		plan.request.Container.GatewayPort = h.nextAvailablePort(plan.request.Container.NetworkMode)
		return nil
	}
	if plan.bridgeMode {
		return nil
	}
	for _, entry := range entries {
		if entry.Port == plan.request.Container.GatewayPort && entry.Name != plan.request.Container.ContainerName {
			return &openClawProvisionFailure{
				message: fmt.Sprintf("Port %d already used by container %q", plan.request.Container.GatewayPort, entry.Name),
				status:  http.StatusConflict,
			}
		}
	}
	if !isTCPPortAvailable(plan.request.Container.GatewayPort) {
		return &openClawProvisionFailure{
			message: fmt.Sprintf(
				"Port %d is already in use on host. Stop the existing gateway/container (e.g. `openclaw gateway stop`) or choose another port.",
				plan.request.Container.GatewayPort,
			),
			status: http.StatusConflict,
		}
	}
	return nil
}

func (h *OpenClawHandler) prepareOpenClawProvisionArtifacts(
	plan *openClawProvisionPlan,
) (openClawProvisionArtifacts, *openClawProvisionFailure) {
	containerDir := h.containerDataDir(plan.request.Container.ContainerName)
	workspaceDir := filepath.Join(containerDir, "workspace")
	if failure := h.writeOpenClawProvisionWorkspace(plan, containerDir, workspaceDir); failure != nil {
		return openClawProvisionArtifacts{}, failure
	}
	configPath := filepath.Join(containerDir, "openclaw.json")
	if err := writeOpenClawConfigForProfile(configPath, plan.request, plan.productionSecurity); err != nil {
		return openClawProvisionArtifacts{}, &openClawProvisionFailure{
			message: "Failed to write private worker config", status: http.StatusInternalServerError,
		}
	}
	absContainerDir, err := filepath.Abs(containerDir)
	if err != nil {
		return openClawProvisionArtifacts{}, &openClawProvisionFailure{
			message: "Failed to resolve private worker directory", status: http.StatusInternalServerError,
		}
	}
	ownerID, err := h.openClawOwnerID()
	if err != nil {
		return openClawProvisionArtifacts{}, &openClawProvisionFailure{
			message: "Failed to establish OpenClaw container ownership", status: http.StatusInternalServerError,
		}
	}
	volumeName := "openclaw-state-" + plan.request.Container.ContainerName
	if err := h.ensureOwnedVolume(volumeName); err != nil {
		return openClawProvisionArtifacts{}, &openClawProvisionFailure{
			message: openClawLifecyclePublicError(err), status: openClawLifecycleHTTPStatus(err),
		}
	}
	return openClawProvisionArtifacts{
		containerDir: absContainerDir,
		workspaceDir: workspaceDir,
		configPath:   configPath,
		ownerID:      ownerID,
	}, nil
}

func (h *OpenClawHandler) writeOpenClawProvisionWorkspace(
	plan *openClawProvisionPlan,
	containerDir string,
	workspaceDir string,
) *openClawProvisionFailure {
	for _, directory := range []string{
		filepath.Join(h.dataDir, "containers"),
		containerDir,
		workspaceDir,
		filepath.Join(workspaceDir, "memory"),
		filepath.Join(workspaceDir, "skills"),
	} {
		if err := ensureOpenClawDirectory(directory, 0o700); err != nil {
			return &openClawProvisionFailure{
				message: "Failed to secure private worker directory", status: http.StatusInternalServerError,
			}
		}
	}
	if err := writeIdentityFiles(workspaceDir, plan.request.Identity); err != nil {
		return &openClawProvisionFailure{message: "Failed to write private identity files", status: http.StatusInternalServerError}
	}
	if err := writePrivateOpenClawFile(filepath.Join(workspaceDir, "AGENTS.md"), []byte(agentsMdContent())); err != nil {
		return &openClawProvisionFailure{
			message: "Failed to write private worker instructions", status: http.StatusInternalServerError,
		}
	}
	for _, skillID := range plan.request.Skills {
		content, err := h.fetchOpenClawSkillContentForProvision(skillID, plan.request.Container.BaseImage)
		if err != nil {
			return &openClawProvisionFailure{message: "Failed to load bounded OpenClaw skill content", status: http.StatusBadRequest}
		}
		if content == "" {
			continue
		}
		skillDir := filepath.Join(workspaceDir, "skills", skillID)
		if err := ensureOpenClawDirectory(skillDir, 0o700); err != nil {
			return &openClawProvisionFailure{
				message: "Failed to secure private skill directory", status: http.StatusInternalServerError,
			}
		}
		if err := writePrivateOpenClawFile(filepath.Join(skillDir, "SKILL.md"), []byte(content)); err != nil {
			return &openClawProvisionFailure{message: "Failed to write private skill", status: http.StatusInternalServerError}
		}
	}
	return nil
}

func (h *OpenClawHandler) startOpenClawProvisionContainer(
	plan *openClawProvisionPlan,
	artifacts openClawProvisionArtifacts,
) (string, *openClawProvisionFailure) {
	attemptLimit := 1
	if !plan.bridgeMode && !plan.requestedPortExplicit {
		attemptLimit = 4
	}
	for attempt := 0; attempt < attemptLimit; attempt++ {
		containerID, retry, failure := h.startOpenClawProvisionAttempt(plan, artifacts, attempt, attemptLimit)
		if failure != nil {
			return "", failure
		}
		if !retry {
			return containerID, nil
		}
	}
	return "", &openClawProvisionFailure{
		message: "Failed to start OpenClaw managed container", status: http.StatusInternalServerError,
	}
}

func (h *OpenClawHandler) startOpenClawProvisionAttempt(
	plan *openClawProvisionPlan,
	artifacts openClawProvisionArtifacts,
	attempt int,
	attemptLimit int,
) (string, bool, *openClawProvisionFailure) {
	if failure := h.prepareOpenClawProvisionStartAttempt(plan, artifacts, attempt, attemptLimit); failure != nil {
		return "", false, failure
	}
	containerID, retry, failure := h.runOpenClawProvisionContainerAttempt(plan, artifacts, attempt, attemptLimit)
	if failure != nil || retry {
		return "", retry, failure
	}
	if plan.bridgeMode || plan.requestedPortExplicit {
		return containerID, false, nil
	}
	return h.resolveOpenClawProvisionImmediateConflict(plan, containerID, attempt, attemptLimit)
}

func (h *OpenClawHandler) prepareOpenClawProvisionStartAttempt(
	plan *openClawProvisionPlan,
	artifacts openClawProvisionArtifacts,
	attempt int,
	attemptLimit int,
) *openClawProvisionFailure {
	if attempt > 0 {
		plan.request.Container.GatewayPort = h.nextAvailablePort(plan.request.Container.NetworkMode)
		if err := writeOpenClawConfigForProfile(artifacts.configPath, plan.request, plan.productionSecurity); err != nil {
			return &openClawProvisionFailure{
				message: "Failed to refresh private worker config", status: http.StatusInternalServerError,
			}
		}
		log.Printf(
			"openclaw: retrying %q with alternate port %d (attempt %d/%d)",
			plan.request.Container.ContainerName,
			plan.request.Container.GatewayPort,
			attempt+1,
			attemptLimit,
		)
	}
	if err := h.removeOwnedContainerIfPresent(plan.request.Container.ContainerName); err != nil {
		return &openClawProvisionFailure{message: openClawLifecyclePublicError(err), status: openClawLifecycleHTTPStatus(err)}
	}
	if plan.productionSecurity {
		if err := h.ensureOpenClawProvisionNetwork(plan.request.Container.NetworkMode, true); err != nil {
			return &openClawProvisionFailure{
				message: "Configured production OpenClaw network is unavailable", status: http.StatusServiceUnavailable,
			}
		}
	}
	return nil
}

func (h *OpenClawHandler) runOpenClawProvisionContainerAttempt(
	plan *openClawProvisionPlan,
	artifacts openClawProvisionArtifacts,
	attempt int,
	attemptLimit int,
) (string, bool, *openClawProvisionFailure) {
	args := openClawContainerRunArgs(plan.request, artifacts.containerDir, artifacts.ownerID)
	commandOutput, err := h.runContainerCommand(args...)
	if err != nil {
		combined := make([]byte, 0, len(commandOutput.stdout)+len(commandOutput.stderr))
		combined = append(combined, commandOutput.stdout...)
		combined = append(combined, commandOutput.stderr...)
		trimmed := strings.TrimSpace(string(combined))
		if isContainerImageMissingError(trimmed) {
			return "", false, &openClawProvisionFailure{
				message: fmt.Sprintf(
					"OpenClaw image %q is unavailable on host runtime. Build or pull this image first, or set OPENCLAW_BASE_IMAGE to an available image before starting dashboard.",
					plan.request.Container.BaseImage,
				),
				status: http.StatusBadRequest,
			}
		}
		canRetry := !plan.bridgeMode && !plan.requestedPortExplicit &&
			isOpenClawGatewayPortConflict(trimmed, plan.request.Container.GatewayPort) && attempt+1 < attemptLimit
		if canRetry {
			log.Printf(
				"openclaw: container runtime start failed due to port conflict on %d; retrying",
				plan.request.Container.GatewayPort,
			)
			return "", true, nil
		}
		return "", false, &openClawProvisionFailure{
			message: "Failed to start OpenClaw managed container", status: http.StatusInternalServerError,
		}
	}
	containerID, err := parseOpenClawContainerID(commandOutput.stdout)
	if err != nil {
		cleanupErr := h.removeOwnedContainerIfPresent(plan.request.Container.ContainerName)
		log.Printf("openclaw: runtime returned invalid container identity cleanup_succeeded=%t", cleanupErr == nil)
		return "", false, &openClawProvisionFailure{
			message: "Failed to verify OpenClaw managed container identity", status: http.StatusInternalServerError,
		}
	}
	if err := h.verifyProvisionedOwnedContainer(plan.request.Container.ContainerName, containerID); err != nil {
		cleanupErr := h.removeProvisionedOwnedContainer(plan.request.Container.ContainerName, containerID)
		log.Printf("openclaw: runtime container ownership verification failed cleanup_succeeded=%t", cleanupErr == nil)
		return "", false, &openClawProvisionFailure{
			message: "Failed to verify OpenClaw managed container ownership", status: http.StatusInternalServerError,
		}
	}
	return containerID, false, nil
}

func (h *OpenClawHandler) resolveOpenClawProvisionImmediateConflict(
	plan *openClawProvisionPlan,
	containerID string,
	attempt int,
	attemptLimit int,
) (string, bool, *openClawProvisionFailure) {
	conflictLogs := h.detectImmediateGatewayPortConflict(
		containerID,
		plan.request.Container.ContainerName,
		plan.request.Container.GatewayPort,
	)
	if conflictLogs == "" {
		return containerID, false, nil
	}
	if err := h.removeProvisionedOwnedContainer(plan.request.Container.ContainerName, containerID); err != nil {
		return "", false, &openClawProvisionFailure{
			message: openClawLifecyclePublicError(err), status: openClawLifecycleHTTPStatus(err),
		}
	}
	if attempt+1 >= attemptLimit {
		return "", false, &openClawProvisionFailure{
			message: "OpenClaw gateway failed to bind an available port", status: http.StatusConflict,
		}
	}
	log.Printf(
		"openclaw: detected gateway port conflict for %q on %d; retrying with a new port",
		plan.request.Container.ContainerName,
		plan.request.Container.GatewayPort,
	)
	return "", true, nil
}

func (h *OpenClawHandler) persistOpenClawProvisionState(
	plan *openClawProvisionPlan,
	state *openClawProvisionState,
	artifacts openClawProvisionArtifacts,
	containerID string,
) *openClawProvisionFailure {
	state.entries = upsertOpenClawProvisionEntry(state.entries, plan.request, artifacts.containerDir, state.teamName)
	teamsChanged := applyOpenClawProvisionLeaderRole(state.entries, state.teams, plan.request)
	sort.Slice(state.entries, func(i, j int) bool { return state.entries[i].Name < state.entries[j].Name })
	if teamsChanged {
		if err := h.saveTeams(state.teams); err != nil {
			cleanupErr := h.removeProvisionedOwnedContainer(plan.request.Container.ContainerName, containerID)
			log.Printf("openclaw: failed to save teams after provisioning cleanup_succeeded=%t", cleanupErr == nil)
			return &openClawProvisionFailure{
				message: "Failed to persist OpenClaw team assignment", status: http.StatusInternalServerError,
			}
		}
	}
	if err := h.saveRegistry(state.entries); err != nil {
		rollbackSucceeded := true
		if teamsChanged {
			rollbackSucceeded = h.saveTeams(state.originalTeams) == nil
		}
		cleanupErr := h.removeProvisionedOwnedContainer(plan.request.Container.ContainerName, containerID)
		log.Printf(
			"openclaw: failed to save registry cleanup_succeeded=%t team_rollback_succeeded=%t",
			cleanupErr == nil,
			rollbackSucceeded,
		)
		return &openClawProvisionFailure{
			message: "Failed to persist OpenClaw worker registry", status: http.StatusInternalServerError,
		}
	}
	return nil
}

func upsertOpenClawProvisionEntry(
	entries []ContainerEntry,
	req ProvisionRequest,
	containerDir string,
	teamName string,
) []ContainerEntry {
	entry := ContainerEntry{
		Name:            req.Container.ContainerName,
		Port:            req.Container.GatewayPort,
		Image:           req.Container.BaseImage,
		Token:           req.Container.AuthToken,
		DataDir:         containerDir,
		CreatedAt:       time.Now().UTC().Format(time.RFC3339),
		TeamID:          req.TeamID,
		TeamName:        teamName,
		AgentName:       strings.TrimSpace(req.Identity.Name),
		AgentEmoji:      strings.TrimSpace(req.Identity.Emoji),
		AgentRole:       strings.TrimSpace(req.Identity.Role),
		AgentVibe:       strings.TrimSpace(req.Identity.Vibe),
		AgentPrinciples: strings.TrimSpace(req.Identity.Principles),
		RoleKind:        req.RoleKind,
	}
	for i := range entries {
		if entries[i].Name == req.Container.ContainerName {
			entry.CreatedAt = entries[i].CreatedAt
			entries[i] = entry
			return entries
		}
	}
	return append(entries, entry)
}

func applyOpenClawProvisionLeaderRole(
	entries []ContainerEntry,
	teams []TeamEntry,
	req ProvisionRequest,
) bool {
	if req.RoleKind != "leader" {
		return false
	}
	for i := range entries {
		if entries[i].TeamID == req.TeamID && entries[i].Name != req.Container.ContainerName {
			entries[i].RoleKind = "worker"
		}
	}
	for i := range teams {
		if teams[i].ID == req.TeamID {
			teams[i].LeaderID = req.Container.ContainerName
			teams[i].UpdatedAt = time.Now().UTC().Format(time.RFC3339)
			return true
		}
	}
	return false
}

func (h *OpenClawHandler) writeOpenClawProvisionSuccess(
	w http.ResponseWriter,
	result *openClawProvisionResult,
) {
	plan := result.plan
	dockerCmd, composeYAML := openClawProvisionGuidance(
		plan.productionSecurity,
		plan.runtimeName,
		plan.request,
		result.artifacts.containerDir,
		result.artifacts.ownerID,
	)
	healthy := false
	for i := 0; i < 10; i++ {
		time.Sleep(2 * time.Second)
		if h.gatewayHealthyForContainer(
			result.containerID,
			plan.request.Container.ContainerName,
			plan.request.Container.GatewayPort,
		) {
			healthy = true
			break
		}
	}
	message := "Container started and gateway is healthy"
	if !healthy {
		message = "Container started but gateway has not become healthy yet (may still be initializing)"
	}
	log.Printf(
		"OpenClaw provisioned: name=%s port=%d healthy=%v",
		plan.request.Container.ContainerName,
		plan.request.Container.GatewayPort,
		healthy,
	)
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(publicOpenClawProvisionResponse(plan.productionSecurity, ProvisionResponse{
		Success:      true,
		Message:      message,
		WorkspaceDir: result.artifacts.workspaceDir,
		ConfigPath:   result.artifacts.configPath,
		ContainerID:  result.containerID,
		DockerCmd:    dockerCmd,
		ComposeYAML:  composeYAML,
	})); err != nil {
		log.Printf("openclaw: provision encode error: %v", err)
	}
}
