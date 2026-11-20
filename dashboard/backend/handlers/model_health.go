package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

const (
	modelHealthPath         = "/api/models/health"
	maxModelHealthModels    = 64
	maxModelHealthNameBytes = 256
)

// The health a model or backend reports. Degraded answers, but slowly, in
// part, or without confirming the model; unchecked had nothing to check.
const (
	ModelHealthStatusHealthy   = "healthy"
	ModelHealthStatusDegraded  = "degraded"
	ModelHealthStatusDown      = "down"
	ModelHealthStatusUnchecked = "unchecked"
)

// ModelHealthResponse is the latest health of configured provider models. It
// names backends as the configuration does and never carries their URLs or
// credentials.
type ModelHealthResponse struct {
	CheckedAt string `json:"checkedAt"`
	// IntervalSeconds is how often a viewer should ask again; answers within
	// it come from the shared cache instead of new probes.
	IntervalSeconds int                `json:"intervalSeconds"`
	Models          []ModelHealthEntry `json:"models"`
}

// ModelHealthEntry is one model's health across its backends.
type ModelHealthEntry struct {
	Model     string `json:"model"`
	Status    string `json:"status"`
	Reason    string `json:"reason,omitempty"`
	CheckedAt string `json:"checkedAt,omitempty"`
	// LatencyMS is the latest answer time of the model's primary answering
	// backend; P95MS is the 95th percentile of RecentMS, the answer times of
	// its recent checks, oldest first.
	LatencyMS *int64                `json:"latencyMs,omitempty"`
	P95MS     *int64                `json:"p95Ms,omitempty"`
	RecentMS  []int64               `json:"recentMs,omitempty"`
	Endpoints []ModelHealthEndpoint `json:"endpoints,omitempty"`
}

// ModelHealthEndpoint is the latest check of one backend of a model.
type ModelHealthEndpoint struct {
	Name       string `json:"name"`
	Status     string `json:"status"`
	Reason     string `json:"reason,omitempty"`
	LatencyMS  *int64 `json:"latencyMs,omitempty"`
	HTTPStatus int    `json:"httpStatus,omitempty"`
}

// ModelHealthErrorResponse explains why no health could be returned.
type ModelHealthErrorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

// ModelHealthHandler reports the health of every configured provider model,
// or of the models named by repeated ?model= parameters. Checks are cheap
// reads (a backend's health path or model list), never inference; results
// are shared between viewers for a short while, and ?refresh=1 asks for a
// new check unless one ran moments ago.
func ModelHealthHandler(configPath string) http.HandlerFunc {
	return newModelHealthHandler(newModelHealthService(configPath, modelHealthOptions{}))
}

func newModelHealthHandler(service *modelHealthService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		if r.URL.Path != modelHealthPath {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			writeModelHealthError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET is supported.")
			return
		}
		query := r.URL.Query()
		models, err := parseModelHealthModels(query["model"])
		if err != nil {
			writeModelHealthError(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}
		refresh := strings.TrimSpace(query.Get("refresh"))
		response, err := service.Check(r.Context(), models, refresh == "1" || strings.EqualFold(refresh, "true"))
		if err != nil {
			var healthErr *modelVerificationError
			if errors.As(err, &healthErr) {
				writeModelHealthError(w, healthErr.status, healthErr.code, healthErr.message)
				return
			}
			writeModelHealthError(w, http.StatusInternalServerError, "health_failed", "Model health could not be checked.")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}
}

func parseModelHealthModels(values []string) ([]string, error) {
	if len(values) > maxModelHealthModels {
		return nil, fmt.Errorf("at most %d models can be checked at once", maxModelHealthModels)
	}
	seen := make(map[string]struct{}, len(values))
	models := make([]string, 0, len(values))
	for _, value := range values {
		model := strings.TrimSpace(value)
		if model == "" {
			return nil, errors.New("model must not be empty")
		}
		if len(model) > maxModelHealthNameBytes {
			return nil, errors.New("model name is too large")
		}
		if strings.ContainsFunc(model, func(character rune) bool { return character < 0x20 || character == 0x7f }) {
			return nil, errors.New("model name contains control characters")
		}
		if _, found := seen[model]; found {
			continue
		}
		seen[model] = struct{}{}
		models = append(models, model)
	}
	return models, nil
}

func writeModelHealthError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(ModelHealthErrorResponse{Error: code, Message: message})
}
