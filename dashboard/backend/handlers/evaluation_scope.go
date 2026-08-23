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
	ResolveEvaluationScope(
		ctx context.Context,
		principal auth.AuthContext,
	) (userIDs []string, teamUsers map[string]string, err error)
}

// EvaluationRunAuthorizer creates one short-lived Router inference credential
// for the authenticated Dashboard principal and the task's Router-owned scope.
// Implementations must never return a stored API-key credential.
type EvaluationRunAuthorizer interface {
	IssueEvaluationInferenceToken(
		ctx context.Context,
		principal auth.AuthContext,
		ownerUserID string,
		ownerTeamID string,
	) (string, error)
}

type evaluationPrincipalScope struct {
	unrestricted bool
	userIDs      map[string]struct{}
	teamUsers    map[string]string
}

func (h *EvaluationHandler) principalScope(r *http.Request) (evaluationPrincipalScope, error) {
	principal, ok := auth.AuthFromContext(r)
	if !ok {
		return evaluationPrincipalScope{unrestricted: true}, nil
	}
	if h.scopeResolver == nil {
		if principal.Perms[auth.PermConfigWrite] {
			return evaluationPrincipalScope{unrestricted: true}, nil
		}
		return evaluationPrincipalScope{}, fmt.Errorf("evaluation scope resolver is not configured")
	}
	userIDs, teamUsers, err := h.scopeResolver.ResolveEvaluationScope(r.Context(), principal)
	if err != nil {
		return evaluationPrincipalScope{}, err
	}
	scope := evaluationPrincipalScope{
		unrestricted: principal.Perms[auth.PermConfigWrite],
		userIDs:      make(map[string]struct{}, len(userIDs)),
		teamUsers:    make(map[string]string, len(teamUsers)),
	}
	for _, userID := range userIDs {
		if userID != "" {
			scope.userIDs[userID] = struct{}{}
		}
	}
	for teamID, userID := range teamUsers {
		if teamID != "" && userID != "" {
			scope.teamUsers[teamID] = userID
		}
	}
	return scope, nil
}

func (scope evaluationPrincipalScope) allows(task *models.EvaluationTask) bool {
	if task == nil {
		return false
	}
	if scope.unrestricted {
		return true
	}
	if _, allowed := scope.userIDs[task.OwnerUserID]; task.OwnerUserID != "" && allowed {
		return true
	}
	_, allowed := scope.teamUsers[task.OwnerTeamID]
	return task.OwnerTeamID != "" && allowed
}

func (scope evaluationPrincipalScope) taskOwner(requestedTeamID string) (string, string, bool) {
	if requestedTeamID != "" {
		userID, allowed := scope.teamUsers[requestedTeamID]
		if !allowed && scope.unrestricted && len(scope.userIDs) == 0 && len(scope.teamUsers) == 0 {
			return "", requestedTeamID, true
		}
		return userID, requestedTeamID, allowed
	}
	if len(scope.teamUsers) == 1 {
		for teamID, userID := range scope.teamUsers {
			return userID, teamID, true
		}
	}
	if len(scope.userIDs) == 1 {
		for userID := range scope.userIDs {
			return userID, "", true
		}
	}
	if scope.unrestricted && len(scope.userIDs) == 0 && len(scope.teamUsers) == 0 {
		return "", "", true
	}
	return "", "", false
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
