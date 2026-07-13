package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"time"
)

// --- Start / Stop / Delete ---

func (h *OpenClawHandler) deleteContainerByName(name string) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	entries, err := h.loadRegistry()
	if err != nil {
		return err
	}
	entryIndex := findContainerIndex(entries, name)
	if entryIndex < 0 {
		return errOpenClawResourceNotFound
	}

	containerInspection, err := h.inspectOpenClawOwnedResource(openClawContainerResource, name)
	if err != nil {
		return err
	}
	volumeName := "openclaw-state-" + name
	volumeInspection, err := h.inspectOpenClawOwnedResource(openClawVolumeResource, volumeName)
	if err != nil {
		return err
	}
	if (containerInspection.exists && !containerInspection.owned) ||
		(volumeInspection.exists && !volumeInspection.owned) {
		return errOpenClawResourceConflict
	}
	if containerInspection.exists {
		if err := h.containerRun("rm", "-f", containerInspection.reference); err != nil {
			return err
		}
	}
	// Volumes do not have an immutable Docker ID. Re-inspect the label owner as
	// close as possible to deletion instead of acting on the stale preflight
	// result after the container removal.
	if volumeInspection.exists {
		if err := h.removeOwnedVolumeIfPresent(volumeName); err != nil {
			return err
		}
	}

	teams, teamErr := h.loadTeams()
	deletedTeamID := entries[entryIndex].TeamID
	entries = append(entries[:entryIndex], entries[entryIndex+1:]...)
	if err := h.saveRegistry(entries); err != nil {
		return err
	}

	if teamErr != nil || deletedTeamID == "" {
		return nil
	}
	teamsChanged := false
	for i := range teams {
		if teams[i].ID == deletedTeamID && teams[i].LeaderID == name {
			teams[i].LeaderID = ""
			teams[i].UpdatedAt = time.Now().UTC().Format(time.RFC3339)
			teamsChanged = true
		}
	}
	if teamsChanged {
		if err := h.saveTeams(teams); err != nil {
			log.Printf("openclaw: failed to clear leader mapping for deleted worker %s", name)
		}
	}
	return nil
}

func (h *OpenClawHandler) runRegisteredContainerLifecycle(name, action string) error {
	_, err := h.runRegisteredOwnedContainerAction(name, action)
	return err
}

func decodeOpenClawContainerName(w http.ResponseWriter, r *http.Request) (string, bool) {
	var req struct {
		ContainerName string `json:"containerName"`
	}
	if status, err := decodeBoundedJSON(w, r, 4*1024, &req); err != nil {
		writeJSONError(w, err.Error(), status)
		return "", false
	}
	rawName := strings.TrimSpace(req.ContainerName)
	name := sanitizeContainerName(rawName)
	if name == "" || name != rawName {
		writeJSONError(w, "containerName must be a canonical OpenClaw worker name", http.StatusBadRequest)
		return "", false
	}
	return name, true
}

func (h *OpenClawHandler) StartHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !h.canManageOpenClaw() {
			h.writeReadOnlyError(w)
			return
		}
		name, ok := decodeOpenClawContainerName(w, r)
		if !ok {
			return
		}
		if err := h.runRegisteredContainerLifecycle(name, "start"); err != nil {
			writeJSONError(w, openClawLifecyclePublicError(err), openClawLifecycleHTTPStatus(err))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": "OpenClaw managed container started",
		}); err != nil {
			log.Printf("openclaw: start encode error: %v", err)
		}
	}
}

func (h *OpenClawHandler) StopHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !h.canManageOpenClaw() {
			h.writeReadOnlyError(w)
			return
		}
		name, ok := decodeOpenClawContainerName(w, r)
		if !ok {
			return
		}
		if err := h.runRegisteredContainerLifecycle(name, "stop"); err != nil {
			writeJSONError(w, openClawLifecyclePublicError(err), openClawLifecycleHTTPStatus(err))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": "OpenClaw managed container stopped",
		}); err != nil {
			log.Printf("openclaw: stop encode error: %v", err)
		}
	}
}

func (h *OpenClawHandler) DeleteHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !h.canManageOpenClaw() {
			h.writeReadOnlyError(w)
			return
		}
		rawName := strings.TrimSpace(strings.TrimPrefix(r.URL.Path, "/api/openclaw/containers/"))
		name := sanitizeContainerName(rawName)
		if name == "" || name != rawName {
			writeJSONError(w, "container name must be canonical", http.StatusBadRequest)
			return
		}

		if err := h.deleteContainerByName(name); err != nil {
			writeJSONError(w, openClawLifecyclePublicError(err), openClawLifecycleHTTPStatus(err))
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": "OpenClaw managed container removed",
		}); err != nil {
			log.Printf("openclaw: delete encode error: %v", err)
		}
	}
}

// --- Dynamic Proxy Lookup ---

// PortForContainer returns the port for a registered container (used by dynamic proxy).
func (h *OpenClawHandler) PortForContainer(name string) (int, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	entries, err := h.loadRegistry()
	if err != nil {
		return 0, false
	}
	for _, e := range entries {
		if e.Name == name {
			return e.Port, true
		}
	}
	return 0, false
}

// TargetBaseForContainer resolves the HTTP base URL for a registered container.
// Uses the container name as hostname (DNS resolution via bridge network).
func (h *OpenClawHandler) TargetBaseForContainer(name string) (string, bool) {
	if dashboardUsesProductionSecurityProfile() {
		resolved, err := h.resolveRegisteredOwnedContainer(name)
		if err != nil {
			return "", false
		}
		return h.gatewayBaseURL(resolved.entry.Name, resolved.entry.Port), true
	}
	port, ok := h.PortForContainer(name)
	if !ok {
		return "", false
	}
	return h.gatewayBaseURL(name, port), true
}
