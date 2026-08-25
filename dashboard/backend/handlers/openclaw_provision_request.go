package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

func (h *OpenClawHandler) handleProvision(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !h.canManageOpenClaw() {
		h.writeReadOnlyError(w)
		return
	}

	asyncRequested := provisionAsyncRequested(r)
	runtimeName, failure := provisionRuntimeName()
	if failure != nil {
		writeJSONError(w, failure.message, failure.status)
		return
	}
	var req ProvisionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"Invalid request: %v"}`, err), http.StatusBadRequest)
		return
	}
	if requestFailure := h.normalizeProvisionRequest(&req); requestFailure != nil {
		writeJSONError(w, requestFailure.message, requestFailure.status)
		return
	}
	if asyncRequested {
		h.queueProvision(w, req)
		return
	}

	requestedPortExplicit := req.Container.GatewayPort != 0
	bridgeMode := isBridgeNetwork(req.Container.NetworkMode)
	outcome, failure := h.runSynchronousProvision(&req, requestedPortExplicit, bridgeMode)
	if failure != nil {
		writeJSONError(w, failure.message, failure.status)
		return
	}
	h.writeProvisionOutcome(w, req, runtimeName, outcome)
}

func provisionRuntimeName() (string, *provisionFailure) {
	runtimeBin, err := detectContainerRuntime()
	if err != nil {
		return "", &provisionFailure{message: err.Error(), status: http.StatusServiceUnavailable}
	}
	runtimeName := filepath.Base(runtimeBin)
	if runtimeName == "" {
		runtimeName = runtimeBin
	}
	return runtimeName, nil
}

func (h *OpenClawHandler) normalizeProvisionRequest(req *ProvisionRequest) *provisionFailure {
	req.Container.ContainerName = deriveContainerName(req.Container.ContainerName, req.Identity.Name)
	if req.Container.AuthToken == "" {
		req.Container.AuthToken = h.gatewayTokenForContainer(req.Container.ContainerName)
		if req.Container.AuthToken == "" {
			req.Container.AuthToken = generateToken(24)
		}
	}
	req.Container.BaseImage = h.resolveBaseImage(req.Container.BaseImage)
	h.normalizeProvisionNetwork(req)
	if req.Container.ModelAPIKey == "" {
		req.Container.ModelAPIKey = "not-needed"
	}
	if strings.TrimSpace(req.Container.ModelName) == "" {
		return &provisionFailure{
			message: "modelName is required; select an active Router entrypoint",
			status:  http.StatusBadRequest,
		}
	}
	if req.Container.MemoryBackend == "" {
		req.Container.MemoryBackend = "local"
	}
	req.TeamID = sanitizeTeamID(req.TeamID)
	if req.TeamID == "" {
		return &provisionFailure{
			message: "teamId is required; create/select a team before provisioning",
			status:  http.StatusBadRequest,
		}
	}
	req.RoleKind = normalizeRoleKind(req.RoleKind)
	return nil
}

func (h *OpenClawHandler) normalizeProvisionNetwork(req *ProvisionRequest) {
	preferred := strings.TrimSpace(os.Getenv("OPENCLAW_DEFAULT_NETWORK_MODE"))
	networkMode := strings.ToLower(strings.TrimSpace(req.Container.NetworkMode))
	if preferred != "" && (networkMode == "" || networkMode == "host" || networkMode == "bridge") {
		req.Container.NetworkMode = preferred
	}
	if req.Container.NetworkMode == "" {
		req.Container.NetworkMode = "host"
	}
	if req.Container.NetworkMode == "host" || strings.HasPrefix(req.Container.NetworkMode, "container:") {
		if req.Container.ModelBaseURL == "" {
			req.Container.ModelBaseURL = h.resolveOpenClawModelBaseURL()
		}
		return
	}
	modelGatewayContainer := openClawModelGatewayContainerName()
	if req.Container.ModelBaseURL == "" {
		req.Container.ModelBaseURL = h.resolveOpenClawModelBaseURL()
	}
	req.Container.ModelBaseURL = rewriteLoopbackHost(req.Container.ModelBaseURL, modelGatewayContainer)
	if req.Container.MemoryBaseURL != "" {
		req.Container.MemoryBaseURL = rewriteLoopbackHost(req.Container.MemoryBaseURL, modelGatewayContainer)
	}
}

func (h *OpenClawHandler) queueProvision(w http.ResponseWriter, req ProvisionRequest) {
	go h.runProvisionAsync(req)
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
}
