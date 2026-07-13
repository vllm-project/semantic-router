package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"time"
)

// --- Provision ---

func (h *OpenClawHandler) ProvisionHandler() http.HandlerFunc {
	return h.handleProvisionRequest
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
		"openclaw: async provision failed for %s (status=%d)",
		req.Container.ContainerName,
		recorder.Code,
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

func (h *OpenClawHandler) detectImmediateGatewayPortConflict(
	containerReference string,
	containerName string,
	port int,
) string {
	for i := 0; i < 80; i++ {
		time.Sleep(500 * time.Millisecond)
		logsOut, err := h.containerCombinedOutput("logs", "--tail", "120", containerReference)
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

func (h *OpenClawHandler) gatewayHealthyForContainer(
	containerReference string,
	containerName string,
	port int,
) bool {
	if !h.containerRunning(containerReference) {
		return false
	}

	logsOut, err := h.containerCombinedOutput("logs", "--tail", "120", containerReference)
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

func (h *OpenClawHandler) containerRunning(containerReference string) bool {
	if dashboardUsesProductionSecurityProfile() && !openClawContainerIDPattern.MatchString(containerReference) {
		inspection, err := h.inspectOpenClawOwnedResource(openClawContainerResource, containerReference)
		if err != nil || !inspection.exists || !inspection.owned {
			return false
		}
		containerReference = inspection.reference
	}
	out, err := h.containerOutput("inspect", "-f", "{{.State.Running}}", containerReference)
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
