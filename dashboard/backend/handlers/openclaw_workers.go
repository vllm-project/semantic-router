package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"
	"time"
)

type workerIdentityPayload struct {
	Name       string `json:"name"`
	Emoji      string `json:"emoji"`
	Role       string `json:"role"`
	Vibe       string `json:"vibe"`
	Principles string `json:"principles"`
}

type workerUpdatePayload struct {
	TeamID   string                `json:"teamId"`
	RoleKind *string               `json:"roleKind,omitempty"`
	Identity workerIdentityPayload `json:"identity"`
}

func workerIdentityProvided(identity workerIdentityPayload) bool {
	return strings.TrimSpace(identity.Name) != "" ||
		strings.TrimSpace(identity.Emoji) != "" ||
		strings.TrimSpace(identity.Role) != "" ||
		strings.TrimSpace(identity.Vibe) != "" ||
		strings.TrimSpace(identity.Principles) != ""
}

func enrichContainerIdentity(entry ContainerEntry) ContainerEntry {
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
	entry.AgentName = snapshot.Name
	entry.AgentEmoji = snapshot.Emoji
	entry.AgentRole = snapshot.Role
	entry.AgentVibe = snapshot.Vibe
	entry.AgentPrinciples = snapshot.Principles
	entry.RoleKind = normalizeRoleKind(entry.RoleKind)
	return entry
}

func findContainerIndex(entries []ContainerEntry, name string) int {
	for i := range entries {
		if entries[i].Name == name {
			return i
		}
	}
	return -1
}

func (h *OpenClawHandler) WorkersHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			h.mu.RLock()
			entries, err := h.loadRegistry()
			teams, teamErr := h.loadTeams()
			h.mu.RUnlock()
			if err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to load workers: %v", err), http.StatusInternalServerError)
				return
			}
			teamNames := make(map[string]string, len(teams))
			if teamErr == nil {
				for _, team := range teams {
					teamNames[team.ID] = team.Name
				}
			}
			for i := range entries {
				entries[i] = enrichContainerIdentity(entries[i])
				if entries[i].TeamName == "" && entries[i].TeamID != "" {
					if teamName, ok := teamNames[entries[i].TeamID]; ok {
						entries[i].TeamName = teamName
					}
				}
			}
			sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(entries); err != nil {
				log.Printf("openclaw: workers encode error: %v", err)
			}
		case http.MethodPost:
			h.ProvisionHandler().ServeHTTP(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func (h *OpenClawHandler) WorkerByIDHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		name := sanitizeContainerName(strings.TrimPrefix(r.URL.Path, "/api/openclaw/workers/"))
		if name == "" {
			writeJSONError(w, "worker id required in path", http.StatusBadRequest)
			return
		}

		switch r.Method {
		case http.MethodGet:
			h.mu.RLock()
			entries, err := h.loadRegistry()
			teams, teamErr := h.loadTeams()
			h.mu.RUnlock()
			if err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to load workers: %v", err), http.StatusInternalServerError)
				return
			}
			index := findContainerIndex(entries, name)
			if index < 0 {
				writeJSONError(w, "worker not found", http.StatusNotFound)
				return
			}

			entry := enrichContainerIdentity(entries[index])
			if entry.TeamName == "" && entry.TeamID != "" && teamErr == nil {
				for _, team := range teams {
					if team.ID == entry.TeamID {
						entry.TeamName = team.Name
						break
					}
				}
			}
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(entry); err != nil {
				log.Printf("openclaw: worker encode error: %v", err)
			}
		case http.MethodPut, http.MethodPatch:
			if !h.canManageOpenClaw() {
				h.writeReadOnlyError(w)
				return
			}

			var req workerUpdatePayload
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				writeJSONError(w, fmt.Sprintf("invalid request body: %v", err), http.StatusBadRequest)
				return
			}

			h.mu.Lock()
			entries, err := h.loadRegistry()
			if err != nil {
				h.mu.Unlock()
				writeJSONError(w, fmt.Sprintf("Failed to load workers: %v", err), http.StatusInternalServerError)
				return
			}
			teams, teamErr := h.loadTeams()
			index := findContainerIndex(entries, name)
			if index < 0 {
				h.mu.Unlock()
				writeJSONError(w, "worker not found", http.StatusNotFound)
				return
			}

			entry := entries[index]
			originalEntry := entry
			entry.RoleKind = normalizeRoleKind(entry.RoleKind)
			teamID := sanitizeTeamID(req.TeamID)
			if teamID != "" {
				if teamErr != nil {
					h.mu.Unlock()
					writeJSONError(w, fmt.Sprintf("Failed to load teams: %v", teamErr), http.StatusInternalServerError)
					return
				}
				team := findTeamByID(teams, teamID)
				if team == nil {
					h.mu.Unlock()
					writeJSONError(w, fmt.Sprintf("team %q not found", teamID), http.StatusNotFound)
					return
				}
				entry.TeamID = teamID
				entry.TeamName = strings.TrimSpace(team.Name)
			}

			teamsChanged := false
			teamIndexes := make(map[string]int, len(teams))
			for i := range teams {
				teamIndexes[teams[i].ID] = i
			}
			now := time.Now().UTC().Format(time.RFC3339)

			if originalEntry.TeamID != "" && originalEntry.TeamID != entry.TeamID {
				if teamIndex, ok := teamIndexes[originalEntry.TeamID]; ok && teams[teamIndex].LeaderID == originalEntry.Name {
					teams[teamIndex].LeaderID = ""
					teams[teamIndex].UpdatedAt = now
					teamsChanged = true
				}
			}

			nextRoleKind := entry.RoleKind
			if req.RoleKind != nil {
				nextRoleKind = normalizeRoleKind(*req.RoleKind)
			} else if originalEntry.TeamID != entry.TeamID && nextRoleKind == "leader" {
				// Moving a leader across teams without explicit role update defaults to worker.
				nextRoleKind = "worker"
			}

			if nextRoleKind == "leader" {
				if entry.TeamID == "" {
					h.mu.Unlock()
					writeJSONError(w, "leader role requires team assignment", http.StatusBadRequest)
					return
				}
				teamIndex, ok := teamIndexes[entry.TeamID]
				if !ok {
					h.mu.Unlock()
					writeJSONError(w, fmt.Sprintf("team %q not found", entry.TeamID), http.StatusNotFound)
					return
				}
				if teams[teamIndex].LeaderID != entry.Name {
					teams[teamIndex].LeaderID = entry.Name
					teams[teamIndex].UpdatedAt = now
					teamsChanged = true
				}
				for i := range entries {
					if entries[i].TeamID == entry.TeamID && entries[i].Name != entry.Name && normalizeRoleKind(entries[i].RoleKind) != "worker" {
						entries[i].RoleKind = "worker"
					}
				}
			} else if entry.TeamID != "" {
				if teamIndex, ok := teamIndexes[entry.TeamID]; ok && teams[teamIndex].LeaderID == entry.Name {
					teams[teamIndex].LeaderID = ""
					teams[teamIndex].UpdatedAt = now
					teamsChanged = true
				}
			}
			entry.RoleKind = nextRoleKind

			if workerIdentityProvided(req.Identity) {
				entry.AgentName = strings.TrimSpace(req.Identity.Name)
				entry.AgentEmoji = strings.TrimSpace(req.Identity.Emoji)
				entry.AgentRole = strings.TrimSpace(req.Identity.Role)
				entry.AgentVibe = strings.TrimSpace(req.Identity.Vibe)
				entry.AgentPrinciples = strings.TrimSpace(req.Identity.Principles)
			}
			if nextRoleKind == "leader" {
				teamName := strings.TrimSpace(entry.TeamName)
				if teamName == "" {
					if teamIndex, ok := teamIndexes[entry.TeamID]; ok {
						teamName = strings.TrimSpace(teams[teamIndex].Name)
					}
				}
				if teamName == "" {
					teamName = "当前团队"
				}

				if strings.TrimSpace(entry.AgentRole) == "" {
					entry.AgentRole = "Team Leader"
				}
				if strings.TrimSpace(entry.AgentVibe) == "" {
					entry.AgentVibe = "统筹-协作"
				}
				if strings.TrimSpace(entry.AgentPrinciples) == "" {
					entry.AgentPrinciples = fmt.Sprintf(
						"你是 %s 的 leader。将目标拆解为可执行任务，主动使用 @<worker-id> 分派工作，并持续同步进展、风险与阻塞。",
						teamName,
					)
				}
			}

			entries[index] = entry
			if err := h.saveRegistry(entries); err != nil {
				h.mu.Unlock()
				writeJSONError(w, fmt.Sprintf("Failed to save workers: %v", err), http.StatusInternalServerError)
				return
			}
			if teamsChanged {
				if err := h.saveTeams(teams); err != nil {
					h.mu.Unlock()
					writeJSONError(w, fmt.Sprintf("Failed to save teams: %v", err), http.StatusInternalServerError)
					return
				}
			}
			h.mu.Unlock()

			updated := enrichContainerIdentity(entry)
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(updated); err != nil {
				log.Printf("openclaw: update worker encode error: %v", err)
			}
		case http.MethodDelete:
			if !h.canManageOpenClaw() {
				h.writeReadOnlyError(w)
				return
			}
			if err := h.deleteContainerByName(name); err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to delete worker: %v", err), http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(map[string]interface{}{
				"success": true,
				"message": fmt.Sprintf("Worker %s removed", name),
			}); err != nil {
				log.Printf("openclaw: delete worker encode error: %v", err)
			}
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}
