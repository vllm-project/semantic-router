package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"
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
			if h.readOnly {
				http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
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

			if workerIdentityProvided(req.Identity) {
				entry.AgentName = strings.TrimSpace(req.Identity.Name)
				entry.AgentEmoji = strings.TrimSpace(req.Identity.Emoji)
				entry.AgentRole = strings.TrimSpace(req.Identity.Role)
				entry.AgentVibe = strings.TrimSpace(req.Identity.Vibe)
				entry.AgentPrinciples = strings.TrimSpace(req.Identity.Principles)
			}

			entries[index] = entry
			if err := h.saveRegistry(entries); err != nil {
				h.mu.Unlock()
				writeJSONError(w, fmt.Sprintf("Failed to save workers: %v", err), http.StatusInternalServerError)
				return
			}
			h.mu.Unlock()

			updated := enrichContainerIdentity(entry)
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(updated); err != nil {
				log.Printf("openclaw: update worker encode error: %v", err)
			}
		case http.MethodDelete:
			if h.readOnly {
				http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
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
