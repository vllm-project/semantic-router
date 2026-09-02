package handlers

import (
	"net/http"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

func (h *EvaluationPlaneHandler) evaluationActor(
	w http.ResponseWriter,
	r *http.Request,
) (evaluationplane.Actor, bool) {
	authContext, ok := dashboardauth.AuthFromContext(r)
	if !ok || authContext.UserID == "" {
		writeEvaluationJSON(w, http.StatusUnauthorized, map[string]any{
			"error": map[string]string{"message": "Authentication is required"},
		})
		return evaluationplane.Actor{}, false
	}
	actor, err := evaluationplane.NewActor(authContext.UserID, authContext.Role == dashboardauth.RoleAdmin)
	if err != nil {
		writeEvaluationJSON(w, http.StatusUnauthorized, map[string]any{
			"error": map[string]string{"message": "Authenticated principal is invalid"},
		})
		return evaluationplane.Actor{}, false
	}
	return actor, true
}
