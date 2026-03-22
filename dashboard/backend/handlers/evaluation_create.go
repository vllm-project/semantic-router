package handlers

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

// validateEvaluationCreateRequest returns (message, status) when invalid; empty message means OK.
func validateEvaluationCreateRequest(req *models.CreateTaskRequest) (string, int) {
	if req.Name == "" {
		return "Task name is required", http.StatusBadRequest
	}
	if len(req.Config.Dimensions) == 0 {
		return "At least one evaluation dimension is required", http.StatusBadRequest
	}
	if req.Config.Level == "" {
		return "Evaluation level is required (router or mom)", http.StatusBadRequest
	}
	if req.Config.Level != models.LevelRouter && req.Config.Level != models.LevelMoM {
		return "Invalid evaluation level. Must be 'router' or 'mom'", http.StatusBadRequest
	}
	for _, dim := range req.Config.Dimensions {
		if msg, code := validateDimensionForLevel(req.Config.Level, dim); msg != "" {
			return msg, code
		}
	}
	return "", 0
}

func validateDimensionForLevel(level models.EvaluationLevel, dim models.EvaluationDimension) (string, int) {
	if level == models.LevelRouter {
		if dim != models.DimensionDomain && dim != models.DimensionFactCheck && dim != models.DimensionUserFeedback {
			return fmt.Sprintf("Dimension '%s' is not valid for router-level evaluation", dim), http.StatusBadRequest
		}
		return "", 0
	}
	if dim == models.DimensionDomain || dim == models.DimensionFactCheck || dim == models.DimensionUserFeedback {
		return fmt.Sprintf("Dimension '%s' is not valid for mom-level evaluation", dim), http.StatusBadRequest
	}
	if dim != models.DimensionAccuracy {
		return fmt.Sprintf("Unknown system dimension '%s'", dim), http.StatusBadRequest
	}
	return "", 0
}

func (h *EvaluationHandler) applyEvaluationCreateDefaults(cfg *models.EvaluationConfig) {
	if cfg.MaxSamples <= 0 {
		cfg.MaxSamples = 50
	}
	if cfg.Endpoint == "" {
		cfg.Endpoint = h.defaultEvaluationEndpoint(cfg.Level)
	}
	if cfg.SamplesPerCat <= 0 {
		cfg.SamplesPerCat = 10
	}
	if cfg.Concurrent <= 0 {
		cfg.Concurrent = 1
	}
}

func (h *EvaluationHandler) defaultEvaluationEndpoint(level models.EvaluationLevel) string {
	if level == models.LevelRouter {
		if h.routerAPIURL != "" {
			return strings.TrimSuffix(h.routerAPIURL, "/") + "/api/v1/eval"
		}
		return "http://localhost:8080/api/v1/eval"
	}
	if h.envoyURL != "" {
		return h.envoyURL
	}
	return "http://localhost:8801"
}
