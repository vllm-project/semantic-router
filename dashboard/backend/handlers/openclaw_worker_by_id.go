package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
)

type workerUpdateFailure struct {
	message    string
	status     int
	logMessage string
}

func (h *OpenClawHandler) handleWorkerByID(w http.ResponseWriter, r *http.Request) {
	rawName := strings.TrimSpace(strings.TrimPrefix(r.URL.Path, "/api/openclaw/workers/"))
	name := sanitizeContainerName(rawName)
	if name == "" || name != rawName {
		writeJSONError(w, "worker id must be one canonical OpenClaw worker name", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodGet:
		h.writeWorkerByID(w, name)
	case http.MethodPut, http.MethodPatch:
		h.updateWorkerByID(w, r, name)
	case http.MethodDelete:
		h.deleteWorkerByID(w, name)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (h *OpenClawHandler) writeWorkerByID(w http.ResponseWriter, name string) {
	h.mu.RLock()
	entries, err := h.loadRegistry()
	teams, teamErr := h.loadTeams()
	h.mu.RUnlock()
	if err != nil {
		log.Printf("openclaw: worker registry load failed")
		writeJSONError(w, "Failed to load OpenClaw workers", http.StatusInternalServerError)
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
}

func (h *OpenClawHandler) updateWorkerByID(w http.ResponseWriter, r *http.Request, name string) {
	if !h.canManageOpenClaw() {
		h.writeReadOnlyError(w)
		return
	}
	var req workerUpdatePayload
	if status, err := decodeBoundedJSON(w, r, 64*1024, &req); err != nil {
		writeJSONError(w, err.Error(), status)
		return
	}

	entry, failure := h.applyWorkerUpdate(name, req)
	if failure != nil {
		if failure.logMessage != "" {
			log.Print(failure.logMessage)
		}
		writeJSONError(w, failure.message, failure.status)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(enrichContainerIdentity(entry)); err != nil {
		log.Printf("openclaw: update worker encode error: %v", err)
	}
}

func (h *OpenClawHandler) applyWorkerUpdate(name string, req workerUpdatePayload) (ContainerEntry, *workerUpdateFailure) {
	h.mu.Lock()
	defer h.mu.Unlock()

	entries, err := h.loadRegistry()
	if err != nil {
		return ContainerEntry{}, &workerUpdateFailure{
			message: "Failed to load OpenClaw workers", status: http.StatusInternalServerError,
			logMessage: "openclaw: worker update registry load failed",
		}
	}
	teams, teamErr := h.loadTeams()
	index := findContainerIndex(entries, name)
	if index < 0 {
		return ContainerEntry{}, &workerUpdateFailure{message: "worker not found", status: http.StatusNotFound}
	}

	entry, teamsChanged, failure := prepareWorkerUpdate(entries, teams, teamErr, index, req)
	if failure != nil {
		return ContainerEntry{}, failure
	}
	entries[index] = entry
	if err := h.saveRegistry(entries); err != nil {
		return ContainerEntry{}, &workerUpdateFailure{
			message: "Failed to save OpenClaw workers", status: http.StatusInternalServerError,
			logMessage: "openclaw: worker registry save failed",
		}
	}
	if teamsChanged {
		if err := h.saveTeams(teams); err != nil {
			return ContainerEntry{}, &workerUpdateFailure{
				message: "Failed to save OpenClaw teams", status: http.StatusInternalServerError,
				logMessage: "openclaw: worker team registry save failed",
			}
		}
	}
	return entry, nil
}

func prepareWorkerUpdate(
	entries []ContainerEntry,
	teams []TeamEntry,
	teamErr error,
	index int,
	req workerUpdatePayload,
) (ContainerEntry, bool, *workerUpdateFailure) {
	originalEntry := entries[index]
	entry := originalEntry
	entry.RoleKind = normalizeRoleKind(entry.RoleKind)
	var failure *workerUpdateFailure
	entry, failure = assignWorkerUpdateTeam(entry, req.TeamID, teams, teamErr)
	if failure != nil {
		return ContainerEntry{}, false, failure
	}

	teamIndexes := make(map[string]int, len(teams))
	for i := range teams {
		teamIndexes[teams[i].ID] = i
	}
	roleKind, teamsChanged, failure := updateWorkerLeadership(
		entry,
		originalEntry,
		req.RoleKind,
		entries,
		teams,
		teamIndexes,
		time.Now().UTC().Format(time.RFC3339),
	)
	if failure != nil {
		return ContainerEntry{}, false, failure
	}
	entry.RoleKind = roleKind
	applyWorkerIdentityUpdate(&entry, req.Identity, teams, teamIndexes)
	return entry, teamsChanged, nil
}

func assignWorkerUpdateTeam(
	entry ContainerEntry,
	rawTeamID string,
	teams []TeamEntry,
	teamErr error,
) (ContainerEntry, *workerUpdateFailure) {
	rawTeamID = strings.TrimSpace(rawTeamID)
	teamID := sanitizeTeamID(rawTeamID)
	if rawTeamID != "" && teamID != rawTeamID {
		return ContainerEntry{}, &workerUpdateFailure{
			message: "teamId must be one canonical OpenClaw team name", status: http.StatusBadRequest,
		}
	}
	if teamID == "" {
		return entry, nil
	}
	if teamErr != nil {
		return ContainerEntry{}, &workerUpdateFailure{
			message: "Failed to load OpenClaw teams", status: http.StatusInternalServerError,
			logMessage: "openclaw: worker update team registry load failed",
		}
	}
	team := findTeamByID(teams, teamID)
	if team == nil {
		return ContainerEntry{}, &workerUpdateFailure{
			message: fmt.Sprintf("team %q not found", teamID), status: http.StatusNotFound,
		}
	}
	entry.TeamID = teamID
	entry.TeamName = strings.TrimSpace(team.Name)
	return entry, nil
}

func updateWorkerLeadership(
	entry ContainerEntry,
	originalEntry ContainerEntry,
	requestedRole *string,
	entries []ContainerEntry,
	teams []TeamEntry,
	teamIndexes map[string]int,
	now string,
) (string, bool, *workerUpdateFailure) {
	teamsChanged := false
	if originalEntry.TeamID != "" && originalEntry.TeamID != entry.TeamID {
		if teamIndex, ok := teamIndexes[originalEntry.TeamID]; ok && teams[teamIndex].LeaderID == originalEntry.Name {
			teams[teamIndex].LeaderID = ""
			teams[teamIndex].UpdatedAt = now
			teamsChanged = true
		}
	}

	nextRoleKind := entry.RoleKind
	if requestedRole != nil {
		nextRoleKind = normalizeRoleKind(*requestedRole)
	} else if originalEntry.TeamID != entry.TeamID && nextRoleKind == "leader" {
		nextRoleKind = "worker"
	}
	if nextRoleKind != "leader" {
		if teamIndex, ok := teamIndexes[entry.TeamID]; entry.TeamID != "" && ok && teams[teamIndex].LeaderID == entry.Name {
			teams[teamIndex].LeaderID = ""
			teams[teamIndex].UpdatedAt = now
			teamsChanged = true
		}
		return nextRoleKind, teamsChanged, nil
	}
	if entry.TeamID == "" {
		return "", false, &workerUpdateFailure{message: "leader role requires team assignment", status: http.StatusBadRequest}
	}
	teamIndex, ok := teamIndexes[entry.TeamID]
	if !ok {
		return "", false, &workerUpdateFailure{
			message: fmt.Sprintf("team %q not found", entry.TeamID), status: http.StatusNotFound,
		}
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
	return nextRoleKind, teamsChanged, nil
}

func applyWorkerIdentityUpdate(
	entry *ContainerEntry,
	identity workerIdentityPayload,
	teams []TeamEntry,
	teamIndexes map[string]int,
) {
	if workerIdentityProvided(identity) {
		entry.AgentName = strings.TrimSpace(identity.Name)
		entry.AgentEmoji = strings.TrimSpace(identity.Emoji)
		entry.AgentRole = strings.TrimSpace(identity.Role)
		entry.AgentVibe = strings.TrimSpace(identity.Vibe)
		entry.AgentPrinciples = strings.TrimSpace(identity.Principles)
	}
	if entry.RoleKind != "leader" {
		return
	}
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

func (h *OpenClawHandler) deleteWorkerByID(w http.ResponseWriter, name string) {
	if !h.canManageOpenClaw() {
		h.writeReadOnlyError(w)
		return
	}
	if err := h.deleteContainerByName(name); err != nil {
		writeJSONError(w, openClawLifecyclePublicError(err), openClawLifecycleHTTPStatus(err))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": fmt.Sprintf("Worker %s removed", name),
	}); err != nil {
		log.Printf("openclaw: delete worker encode error: %v", err)
	}
}
