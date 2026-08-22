package handlers

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

// EvaluationScopeResolver provides the model-serving teams attached to a
// Dashboard identity. Dashboard roles decide the operation; this scope decides
// which evaluation data the identity may see.
type EvaluationScopeResolver interface {
	ListTeamIDsForUser(ctx context.Context, userID string) ([]string, error)
}

type evaluationPrincipalScope struct {
	unrestricted bool
	userID       string
	teamIDs      map[string]struct{}
}

func (h *EvaluationHandler) principalScope(r *http.Request) (evaluationPrincipalScope, error) {
	principal, ok := auth.AuthFromContext(r)
	if !ok || principal.Perms[auth.PermConfigWrite] {
		return evaluationPrincipalScope{unrestricted: true}, nil
	}
	if h.scopeResolver == nil {
		return evaluationPrincipalScope{}, fmt.Errorf("evaluation scope resolver is not configured")
	}
	teamIDs, err := h.scopeResolver.ListTeamIDsForUser(r.Context(), principal.UserID)
	if err != nil {
		return evaluationPrincipalScope{}, err
	}
	scope := evaluationPrincipalScope{
		userID:  principal.UserID,
		teamIDs: make(map[string]struct{}, len(teamIDs)),
	}
	for _, teamID := range teamIDs {
		scope.teamIDs[teamID] = struct{}{}
	}
	return scope, nil
}

func (scope evaluationPrincipalScope) allows(task *models.EvaluationTask) bool {
	if task == nil {
		return false
	}
	if scope.unrestricted || (scope.userID != "" && task.OwnerUserID == scope.userID) {
		return true
	}
	_, allowed := scope.teamIDs[task.OwnerTeamID]
	return task.OwnerTeamID != "" && allowed
}

func (h *EvaluationHandler) authorizedTask(
	w http.ResponseWriter,
	r *http.Request,
	taskID string,
) (*models.EvaluationTask, bool) {
	scope, err := h.principalScope(r)
	if err != nil {
		http.Error(w, "Evaluation scope is unavailable", http.StatusServiceUnavailable)
		return nil, false
	}
	task, err := h.db.GetTask(taskID)
	if err != nil {
		log.Printf("Failed to get task: %v", err)
		http.Error(w, "Failed to get task", http.StatusInternalServerError)
		return nil, false
	}
	if !scope.allows(task) {
		http.Error(w, "Task not found", http.StatusNotFound)
		return nil, false
	}
	return task, true
}
