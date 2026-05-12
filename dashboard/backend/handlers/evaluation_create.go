package handlers

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluation"
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
	if msg, code := validateDatasetsForConfig(req.Config); msg != "" {
		return msg, code
	}
	return "", 0
}

func normalizeEvaluationCreateConfig(cfg *models.EvaluationConfig) {
	if cfg.Datasets == nil {
		cfg.Datasets = map[string][]string{}
		return
	}

	normalizedDatasets := make(map[string][]string, len(cfg.Dimensions))
	for _, dim := range cfg.Dimensions {
		normalizedDatasets[string(dim)] = normalizeDatasetNames(cfg.Datasets[string(dim)])
	}

	cfg.Datasets = normalizedDatasets
}

func normalizeDatasetNames(datasets []string) []string {
	cleaned := make([]string, 0, len(datasets))
	seen := make(map[string]struct{}, len(datasets))

	for _, dataset := range datasets {
		dataset = strings.TrimSpace(dataset)
		if dataset == "" || strings.EqualFold(dataset, "default") {
			continue
		}
		if _, ok := seen[dataset]; ok {
			continue
		}
		seen[dataset] = struct{}{}
		cleaned = append(cleaned, dataset)
	}

	return cleaned
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

func validateDatasetsForConfig(cfg models.EvaluationConfig) (string, int) {
	availableDatasets := evaluation.GetAvailableDatasets()

	for _, dim := range cfg.Dimensions {
		requestedDatasets := cfg.Datasets[string(dim)]
		if len(requestedDatasets) == 0 {
			continue
		}

		allowedDatasets := make(map[string]struct{})
		for _, dataset := range availableDatasets[string(dim)] {
			if dataset.Level == cfg.Level && dataset.Dimension == dim {
				allowedDatasets[dataset.Name] = struct{}{}
			}
		}

		for _, dataset := range requestedDatasets {
			if _, ok := allowedDatasets[dataset]; !ok {
				return fmt.Sprintf(
					"Dataset '%s' is not valid for %s dimension at %s-level evaluation",
					dataset,
					dim,
					cfg.Level,
				), http.StatusBadRequest
			}
		}
	}

	return "", 0
}

// fallbackRouterEvalEndpoint is the placeholder router eval URL the dashboard
// frontend pre-populates when no in-cluster router address is available. The
// frontend always sends this value, so the backend treats it as "unset" when a
// server-side TARGET_ROUTER_API_URL override is configured.
const fallbackRouterEvalEndpoint = "http://localhost:8080/api/v1/eval"

func (h *EvaluationHandler) applyEvaluationCreateDefaults(cfg *models.EvaluationConfig) {
	if cfg.MaxSamples <= 0 {
		cfg.MaxSamples = 50
	}
	if cfg.Endpoint == "" || h.shouldOverrideEndpoint(cfg) {
		cfg.Endpoint = h.defaultEvaluationEndpoint(cfg.Level)
	}
	if cfg.SamplesPerCat <= 0 {
		cfg.SamplesPerCat = 10
	}
	if cfg.Concurrent <= 0 {
		cfg.Concurrent = 1
	}
}

// shouldOverrideEndpoint returns true when the incoming endpoint matches the
// frontend placeholder default and a server-side override is configured. This
// keeps TARGET_ROUTER_API_URL effective in containerized deployments where the
// dashboard cannot reach localhost:8080.
func (h *EvaluationHandler) shouldOverrideEndpoint(cfg *models.EvaluationConfig) bool {
	if cfg.Level != models.LevelRouter {
		return false
	}
	if h.routerAPIURL == "" {
		return false
	}
	return cfg.Endpoint == fallbackRouterEvalEndpoint
}

func (h *EvaluationHandler) defaultEvaluationEndpoint(level models.EvaluationLevel) string {
	if level == models.LevelRouter {
		if h.routerAPIURL != "" {
			return strings.TrimSuffix(h.routerAPIURL, "/") + "/api/v1/eval"
		}
		return fallbackRouterEvalEndpoint
	}
	if h.envoyURL != "" {
		return h.envoyURL
	}
	return "http://localhost:8801"
}
