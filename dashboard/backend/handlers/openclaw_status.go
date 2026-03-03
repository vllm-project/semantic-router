package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// --- Token ---

func (h *OpenClawHandler) gatewayTokenForContainer(name string) string {
	entry := h.findEntry(name)
	if entry != nil && entry.Token != "" {
		return entry.Token
	}
	configPath := filepath.Join(h.containerDataDir(name), "openclaw.json")
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
	return cfg.Gateway.Auth.Token
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
			"token": h.gatewayTokenForContainer(name),
		}); err != nil {
			log.Printf("openclaw: token encode error: %v", err)
		}
	}
}

// --- Status ---

func (h *OpenClawHandler) gatewayHostCandidates() []string {
	candidates := []string{}

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

func (h *OpenClawHandler) resolveGatewayHost(port int) string {
	candidates := h.gatewayHostCandidates()
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

func (h *OpenClawHandler) gatewayBaseURL(port int) string {
	host := h.resolveGatewayHost(port)
	return fmt.Sprintf("http://%s:%d", host, port)
}

func (h *OpenClawHandler) gatewayReachable(port int) bool {
	client := &http.Client{Timeout: 1200 * time.Millisecond}
	for _, host := range h.gatewayHostCandidates() {
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

func (h *OpenClawHandler) checkContainerHealth(entry ContainerEntry) OpenClawStatus {
	snapshot := identitySnapshot{
		Name:       entry.AgentName,
		Emoji:      entry.AgentEmoji,
		Role:       entry.AgentRole,
		Vibe:       entry.AgentVibe,
		Principles: entry.AgentPrinciples,
	}
	if (snapshot.Name == "" || snapshot.Role == "" || snapshot.Vibe == "" || snapshot.Principles == "") && entry.DataDir != "" {
		fileSnapshot := readIdentitySnapshot(entry.DataDir)
		if snapshot.Name == "" {
			snapshot.Name = fileSnapshot.Name
		}
		if snapshot.Emoji == "" {
			snapshot.Emoji = fileSnapshot.Emoji
		}
		if snapshot.Role == "" {
			snapshot.Role = fileSnapshot.Role
		}
		if snapshot.Vibe == "" {
			snapshot.Vibe = fileSnapshot.Vibe
		}
		if snapshot.Principles == "" {
			snapshot.Principles = fileSnapshot.Principles
		}
	}

	status := OpenClawStatus{
		ContainerName:   entry.Name,
		GatewayURL:      h.gatewayBaseURL(entry.Port),
		Port:            entry.Port,
		Image:           entry.Image,
		CreatedAt:       entry.CreatedAt,
		TeamID:          entry.TeamID,
		TeamName:        entry.TeamName,
		AgentName:       snapshot.Name,
		AgentEmoji:      snapshot.Emoji,
		AgentRole:       snapshot.Role,
		AgentVibe:       snapshot.Vibe,
		AgentPrinciples: snapshot.Principles,
	}

	out, err := h.containerOutput("inspect", "-f", "{{.State.Running}}", entry.Name)
	if err != nil {
		status.Running = false
		if strings.Contains(err.Error(), "container runtime not available") {
			status.Error = err.Error()
			return status
		}
		status.Error = "Container not found"
		return status
	}
	status.Running = strings.TrimSpace(string(out)) == "true"
	if !status.Running {
		status.Error = "Container stopped"
		return status
	}

	gatewayReachable := h.gatewayReachable(entry.Port)
	if !gatewayReachable {
		status.Error = "Gateway not reachable"
		return status
	}

	// Compare positions so a successful restart after a previous failure is correctly detected.
	logOut, logErr := h.containerCombinedOutput("logs", "--tail", "80", entry.Name)
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
			writeJSONError(w, fmt.Sprintf("Failed to load registry: %v", err), http.StatusInternalServerError)
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
		h.mu.RLock()
		port := h.nextAvailablePort()
		h.mu.RUnlock()
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]int{"port": port}); err != nil {
			log.Printf("openclaw: next-port encode error: %v", err)
		}
	}
}
