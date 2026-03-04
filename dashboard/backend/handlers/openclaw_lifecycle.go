package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
)

// --- Start / Stop / Delete ---

func (h *OpenClawHandler) deleteContainerByName(name string) error {
	_ = h.containerRun("rm", "-f", name)
	_ = h.containerRun("volume", "rm", "openclaw-state-"+name)

	h.mu.Lock()
	defer h.mu.Unlock()

	entries, err := h.loadRegistry()
	if err != nil {
		return err
	}
	teams, teamErr := h.loadTeams()
	filtered := entries[:0]
	deletedTeamID := ""
	for _, e := range entries {
		if e.Name != name {
			filtered = append(filtered, e)
			continue
		}
		deletedTeamID = e.TeamID
	}
	if err := h.saveRegistry(filtered); err != nil {
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
			log.Printf("openclaw: failed to clear leader mapping for deleted worker %s: %v", name, err)
		}
	}
	return nil
}

func (h *OpenClawHandler) StartHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if h.readOnly {
			http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
			return
		}
		var req struct {
			ContainerName string `json:"containerName"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, "invalid request body", http.StatusBadRequest)
			return
		}
		if req.ContainerName == "" {
			writeJSONError(w, "containerName required", http.StatusBadRequest)
			return
		}
		out, err := h.containerCombinedOutput("start", req.ContainerName)
		if err != nil {
			writeJSONError(w, fmt.Sprintf("Failed to start: %s (%v)", strings.TrimSpace(string(out)), err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": fmt.Sprintf("Container %s started", req.ContainerName),
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
		if h.readOnly {
			http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
			return
		}
		var req struct {
			ContainerName string `json:"containerName"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, "invalid request body", http.StatusBadRequest)
			return
		}
		if req.ContainerName == "" {
			writeJSONError(w, "containerName required", http.StatusBadRequest)
			return
		}
		out, err := h.containerCombinedOutput("stop", req.ContainerName)
		if err != nil {
			writeJSONError(w, fmt.Sprintf("Failed to stop: %s (%v)", strings.TrimSpace(string(out)), err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": fmt.Sprintf("Container %s stopped", req.ContainerName),
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
		if h.readOnly {
			http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
			return
		}
		name := strings.TrimPrefix(r.URL.Path, "/api/openclaw/containers/")
		if name == "" {
			writeJSONError(w, "container name required in path", http.StatusBadRequest)
			return
		}

		if err := h.deleteContainerByName(name); err != nil {
			log.Printf("openclaw: failed to save registry on delete: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": fmt.Sprintf("Container %s removed", name),
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
func (h *OpenClawHandler) TargetBaseForContainer(name string) (string, bool) {
	port, ok := h.PortForContainer(name)
	if !ok {
		return "", false
	}
	return h.gatewayBaseURL(port), true
}
