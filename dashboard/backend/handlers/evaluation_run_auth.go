package handlers

import (
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluation"
	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

func (h *EvaluationHandler) evaluationRunAuthorization(
	w http.ResponseWriter,
	r *http.Request,
	task *models.EvaluationTask,
) (evaluation.InferenceAuthorization, bool) {
	principal, authenticated := auth.AuthFromContext(r)
	if !authenticated || h.runAuthorizer == nil || task == nil {
		http.Error(w, "Evaluation inference authorization is unavailable", http.StatusServiceUnavailable)
		return evaluation.InferenceAuthorization{}, false
	}
	token, err := h.runAuthorizer.IssueEvaluationInferenceToken(
		r.Context(),
		principal,
		task.OwnerUserID,
		task.OwnerTeamID,
	)
	if err != nil {
		http.Error(w, "Evaluation inference authorization is unavailable", http.StatusServiceUnavailable)
		return evaluation.InferenceAuthorization{}, false
	}
	authorization, err := evaluation.NewInferenceAuthorization(token)
	if err != nil {
		http.Error(w, "Evaluation inference authorization is unavailable", http.StatusServiceUnavailable)
		return evaluation.InferenceAuthorization{}, false
	}
	return authorization, true
}
