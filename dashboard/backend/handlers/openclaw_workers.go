package handlers

import (
	"encoding/json"
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
				log.Printf("openclaw: workers registry load failed")
				writeJSONError(w, "Failed to load OpenClaw workers", http.StatusInternalServerError)
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
	return h.handleWorkerByID
}
