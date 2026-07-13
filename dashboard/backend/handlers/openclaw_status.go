package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// --- Token ---

func (h *OpenClawHandler) gatewayTokenFromConfigPath(configPath string) string {
	configPath = strings.TrimSpace(configPath)
	if configPath == "" {
		return ""
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}
	var cfg struct {
		Gateway struct {
			Auth struct {
				Token string `json:"token"`
			} `json:"auth"`
		} `json:"gateway"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return ""
	}
	return strings.TrimSpace(cfg.Gateway.Auth.Token)
}

func (h *OpenClawHandler) gatewayTokenForContainer(name string) string {
	normalizedName := sanitizeContainerName(name)
	if normalizedName == "" {
		return ""
	}

	entry := h.findEntry(normalizedName)
	if entry != nil && strings.TrimSpace(entry.DataDir) != "" {
		configPath := filepath.Join(strings.TrimSpace(entry.DataDir), "openclaw.json")
		if token := h.gatewayTokenFromConfigPath(configPath); token != "" {
			return token
		}
	}

	// Backward-compatible fallback path.
	if token := h.gatewayTokenFromConfigPath(filepath.Join(h.containerDataDir(normalizedName), "openclaw.json")); token != "" {
		return token
	}

	if entry != nil {
		return strings.TrimSpace(entry.Token)
	}
	return ""
}

func (h *OpenClawHandler) GatewayTokenForContainer(name string) string {
	return h.gatewayTokenForContainer(name)
}

func (h *OpenClawHandler) TokenHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		name := r.URL.Query().Get("name")
		if name == "" {
			http.Error(w, `{"error":"name parameter required"}`, http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{
			"token": h.GatewayTokenForContainer(name),
		}); err != nil {
			log.Printf("openclaw: token encode error: %v", err)
		}
	}
}

// --- Status ---

func (h *OpenClawHandler) gatewayHostCandidates(containerName string) []string {
	candidates := []string{}

	// First priority: container name (DNS resolution within bridge network)
	if containerName != "" {
		candidates = append(candidates, containerName)
	}

	if explicit := strings.TrimSpace(os.Getenv("OPENCLAW_GATEWAY_HOST")); explicit != "" {
		candidates = append(candidates, explicit)
	}
	if explicitList := strings.TrimSpace(os.Getenv("OPENCLAW_GATEWAY_HOSTS")); explicitList != "" {
		for _, raw := range strings.Split(explicitList, ",") {
			if host := strings.TrimSpace(raw); host != "" {
				candidates = append(candidates, host)
			}
		}
	}

	candidates = append(candidates,
		"127.0.0.1",
		"host.docker.internal",
		"host.containers.internal",
	)

	seen := map[string]bool{}
	out := make([]string, 0, len(candidates))
	for _, host := range candidates {
		if host == "" || seen[host] {
			continue
		}
		seen[host] = true
		out = append(out, host)
	}
	return out
}

func (h *OpenClawHandler) resolveGatewayHost(containerName string, port int) string {
	candidates := h.gatewayHostCandidates(containerName)
	if len(candidates) == 0 {
		return "127.0.0.1"
	}
	for _, host := range candidates {
		if canConnectTCP(host, port, 350*time.Millisecond) {
			return host
		}
	}
	return candidates[0]
}

func (h *OpenClawHandler) gatewayBaseURL(containerName string, port int) string {
	host := h.resolveGatewayHost(containerName, port)
	return fmt.Sprintf("http://%s:%d", host, port)
}

func (h *OpenClawHandler) gatewayReachable(containerName string, port int) bool {
	client := &http.Client{Timeout: 1200 * time.Millisecond}
	for _, host := range h.gatewayHostCandidates(containerName) {
		target := fmt.Sprintf("http://%s:%d/health", host, port)
		resp, err := client.Get(target)
		if err == nil {
			resp.Body.Close()
			// Any HTTP response confirms the gateway endpoint is reachable.
			return true
		}
		if canConnectTCP(host, port, 350*time.Millisecond) {
			return true
		}
	}
	return false
}

func openClawStatusFromEntry(entry ContainerEntry) OpenClawStatus {
	return OpenClawStatus{
		ContainerName:   entry.Name,
		Port:            entry.Port,
		Image:           entry.Image,
		CreatedAt:       entry.CreatedAt,
		TeamID:          entry.TeamID,
		TeamName:        entry.TeamName,
		AgentName:       entry.AgentName,
		AgentEmoji:      entry.AgentEmoji,
		AgentRole:       entry.AgentRole,
		AgentVibe:       entry.AgentVibe,
		AgentPrinciples: entry.AgentPrinciples,
		RoleKind:        normalizeRoleKind(entry.RoleKind),
	}
}

func (h *OpenClawHandler) checkContainerHealth(entry ContainerEntry) OpenClawStatus {
	status := OpenClawStatus{}
	resolved, resolveErr := h.resolveRegisteredOwnedContainer(entry.Name)
	if resolved.entry.Name != "" {
		entry = enrichContainerIdentity(resolved.entry)
		status = openClawStatusFromEntry(entry)
	}
	if resolveErr != nil {
		switch {
		case errors.Is(resolveErr, errOpenClawResourceConflict):
			status.Error = "Container ownership conflict"
		case errors.Is(resolveErr, errOpenClawResourceNotFound) && resolved.entry.Name == "":
			status.Error = "Container not in registry"
		default:
			status.Error = "Container unavailable"
		}
		return status
	}

	status.GatewayURL = h.gatewayBaseURL(entry.Name, entry.Port)

	out, err := h.containerOutput("inspect", "-f", "{{.State.Running}}", resolved.reference)
	if err != nil {
		status.Running = false
		status.Error = "Container unavailable"
		return status
	}
	status.Running = strings.TrimSpace(string(out)) == "true"
	if !status.Running {
		status.Error = "Container stopped"
		return status
	}

	gatewayReachable := h.gatewayReachable(entry.Name, entry.Port)
	if !gatewayReachable {
		status.Error = "Gateway not reachable"
		return status
	}

	// Compare positions so a successful restart after a previous failure is correctly detected.
	logOut, logErr := h.containerCombinedOutput("logs", "--tail", "80", resolved.reference)
	if logErr == nil {
		logs := string(logOut)
		lastSuccess := strings.LastIndex(logs, "[gateway] listening on ws://")
		lastFail := max(
			strings.LastIndex(logs, "failed to start:"),
			strings.LastIndex(logs, "permission denied, mkdir '/state/"),
		)
		if gatewayReachable && lastSuccess >= 0 && lastSuccess > lastFail {
			status.Healthy = true
		} else if lastFail >= 0 && lastFail > lastSuccess {
			status.Healthy = false
			status.Error = "Subsystem initialization failure"
		} else if gatewayReachable {
			status.Healthy = true
		}
	} else if gatewayReachable {
		status.Healthy = true
	}
	if status.Healthy {
		status.Error = ""
	}
	return status
}

func (h *OpenClawHandler) StatusHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		h.mu.RLock()
		entries, err := h.loadRegistry()
		teams, teamErr := h.loadTeams()
		h.mu.RUnlock()
		if err != nil {
			log.Printf("openclaw: status registry load failed")
			writeJSONError(w, "Failed to load OpenClaw registry", http.StatusInternalServerError)
			return
		}
		teamNames := map[string]string{}
		if teamErr == nil {
			for _, team := range teams {
				teamNames[team.ID] = team.Name
			}
		}

		enrichTeam := func(status *OpenClawStatus) {
			if status == nil || status.TeamID == "" {
				return
			}
			if status.TeamName == "" {
				if name, ok := teamNames[status.TeamID]; ok {
					status.TeamName = name
				}
			}
		}

		name := r.URL.Query().Get("name")
		if name != "" {
			for _, e := range entries {
				if e.Name == name {
					status := h.checkContainerHealth(e)
					enrichTeam(&status)
					w.Header().Set("Content-Type", "application/json")
					if err := json.NewEncoder(w).Encode(status); err != nil {
						log.Printf("openclaw: status encode error: %v", err)
					}
					return
				}
			}
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(OpenClawStatus{Error: "Container not in registry"}); err != nil {
				log.Printf("openclaw: status encode error: %v", err)
			}
			return
		}

		statuses := make([]OpenClawStatus, 0, len(entries))
		for _, e := range entries {
			status := h.checkContainerHealth(e)
			if status.ContainerName == "" {
				continue
			}
			enrichTeam(&status)
			statuses = append(statuses, status)
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(statuses); err != nil {
			log.Printf("openclaw: statuses encode error: %v", err)
		}
	}
}

// --- Next Port ---

func (h *OpenClawHandler) NextPortHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		// Accept optional networkMode query param; defaults to host mode behavior
		networkMode := r.URL.Query().Get("networkMode")
		h.mu.RLock()
		port := h.nextAvailablePort(networkMode)
		h.mu.RUnlock()
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]int{"port": port}); err != nil {
			log.Printf("openclaw: next-port encode error: %v", err)
		}
	}
}
