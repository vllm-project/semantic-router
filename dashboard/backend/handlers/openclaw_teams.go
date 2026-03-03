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

// --- Teams ---

type teamPayload struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Vibe        string `json:"vibe"`
	Role        string `json:"role"`
	Principal   string `json:"principal"`
	Description string `json:"description"`
}

func (h *OpenClawHandler) TeamsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			h.mu.RLock()
			teams, err := h.loadTeams()
			h.mu.RUnlock()
			if err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to load teams: %v", err), http.StatusInternalServerError)
				return
			}
			sort.Slice(teams, func(i, j int) bool { return teams[i].Name < teams[j].Name })
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(teams); err != nil {
				log.Printf("openclaw: teams encode error: %v", err)
			}
		case http.MethodPost:
			if h.readOnly {
				http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
				return
			}

			var req teamPayload
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				writeJSONError(w, fmt.Sprintf("invalid request body: %v", err), http.StatusBadRequest)
				return
			}
			name := strings.TrimSpace(req.Name)
			if name == "" {
				writeJSONError(w, "team name required", http.StatusBadRequest)
				return
			}

			teamID := sanitizeTeamID(req.ID)
			if teamID == "" {
				teamID = sanitizeTeamID(name)
			}
			if teamID == "" {
				writeJSONError(w, "team id is invalid", http.StatusBadRequest)
				return
			}

			h.mu.Lock()
			defer h.mu.Unlock()

			teams, err := h.loadTeams()
			if err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to load teams: %v", err), http.StatusInternalServerError)
				return
			}
			for _, existing := range teams {
				if existing.ID == teamID {
					writeJSONError(w, fmt.Sprintf("team %q already exists", teamID), http.StatusConflict)
					return
				}
				if strings.EqualFold(strings.TrimSpace(existing.Name), name) {
					writeJSONError(w, fmt.Sprintf("team name %q already exists", name), http.StatusConflict)
					return
				}
			}

			now := time.Now().UTC().Format(time.RFC3339)
			created := TeamEntry{
				ID:          teamID,
				Name:        name,
				Vibe:        strings.TrimSpace(req.Vibe),
				Role:        strings.TrimSpace(req.Role),
				Principal:   strings.TrimSpace(req.Principal),
				Description: strings.TrimSpace(req.Description),
				CreatedAt:   now,
				UpdatedAt:   now,
			}
			teams = append(teams, created)
			sort.Slice(teams, func(i, j int) bool { return teams[i].Name < teams[j].Name })
			if err := h.saveTeams(teams); err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to save teams: %v", err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			if err := json.NewEncoder(w).Encode(created); err != nil {
				log.Printf("openclaw: create team encode error: %v", err)
			}
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func (h *OpenClawHandler) TeamByIDHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		teamID := sanitizeTeamID(strings.TrimPrefix(r.URL.Path, "/api/openclaw/teams/"))
		if teamID == "" {
			writeJSONError(w, "team id required in path", http.StatusBadRequest)
			return
		}

		switch r.Method {
		case http.MethodGet:
			h.mu.RLock()
			teams, err := h.loadTeams()
			h.mu.RUnlock()
			if err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to load teams: %v", err), http.StatusInternalServerError)
				return
			}
			team := findTeamByID(teams, teamID)
			if team == nil {
				writeJSONError(w, "team not found", http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(team); err != nil {
				log.Printf("openclaw: team encode error: %v", err)
			}
		case http.MethodPut, http.MethodPatch:
			if h.readOnly {
				http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
				return
			}

			var req teamPayload
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				writeJSONError(w, fmt.Sprintf("invalid request body: %v", err), http.StatusBadRequest)
				return
			}
			name := strings.TrimSpace(req.Name)
			if name == "" {
				writeJSONError(w, "team name required", http.StatusBadRequest)
				return
			}

			h.mu.Lock()
			defer h.mu.Unlock()

			teams, err := h.loadTeams()
			if err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to load teams: %v", err), http.StatusInternalServerError)
				return
			}

			index := -1
			for i := range teams {
				if teams[i].ID == teamID {
					index = i
					continue
				}
				if strings.EqualFold(strings.TrimSpace(teams[i].Name), name) {
					writeJSONError(w, fmt.Sprintf("team name %q already exists", name), http.StatusConflict)
					return
				}
			}
			if index < 0 {
				writeJSONError(w, "team not found", http.StatusNotFound)
				return
			}

			teams[index].Name = name
			teams[index].Vibe = strings.TrimSpace(req.Vibe)
			teams[index].Role = strings.TrimSpace(req.Role)
			teams[index].Principal = strings.TrimSpace(req.Principal)
			teams[index].Description = strings.TrimSpace(req.Description)
			teams[index].UpdatedAt = time.Now().UTC().Format(time.RFC3339)
			updated := teams[index]

			sort.Slice(teams, func(i, j int) bool { return teams[i].Name < teams[j].Name })
			err = h.saveTeams(teams)
			if err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to save teams: %v", err), http.StatusInternalServerError)
				return
			}

			entries, err := h.loadRegistry()
			if err == nil {
				changed := false
				for i := range entries {
					if entries[i].TeamID == teamID {
						entries[i].TeamName = updated.Name
						changed = true
					}
				}
				if changed {
					if err := h.saveRegistry(entries); err != nil {
						log.Printf("openclaw: failed to save registry after team rename: %v", err)
					}
				}
			}

			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(updated); err != nil {
				log.Printf("openclaw: update team encode error: %v", err)
			}
		case http.MethodDelete:
			if h.readOnly {
				http.Error(w, `{"error":"Read-only mode enabled"}`, http.StatusForbidden)
				return
			}

			h.mu.Lock()
			defer h.mu.Unlock()

			entries, err := h.loadRegistry()
			if err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to load registry: %v", err), http.StatusInternalServerError)
				return
			}
			assigned := 0
			for _, e := range entries {
				if e.TeamID == teamID {
					assigned++
				}
			}
			if assigned > 0 {
				writeJSONError(w, fmt.Sprintf("team is still assigned to %d agent(s)", assigned), http.StatusConflict)
				return
			}

			teams, err := h.loadTeams()
			if err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to load teams: %v", err), http.StatusInternalServerError)
				return
			}
			filtered := teams[:0]
			removed := false
			for _, team := range teams {
				if team.ID == teamID {
					removed = true
					continue
				}
				filtered = append(filtered, team)
			}
			if !removed {
				writeJSONError(w, "team not found", http.StatusNotFound)
				return
			}
			if err := h.saveTeams(filtered); err != nil {
				writeJSONError(w, fmt.Sprintf("Failed to save teams: %v", err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(map[string]interface{}{
				"success": true,
				"message": fmt.Sprintf("Team %s removed", teamID),
			}); err != nil {
				log.Printf("openclaw: delete team encode error: %v", err)
			}
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}
