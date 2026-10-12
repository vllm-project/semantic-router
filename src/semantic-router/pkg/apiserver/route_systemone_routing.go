//go:build !windows

package apiserver

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

const systemOneRoutingDiagnosticPath = apiDiagnosticsPath + "/routes/systemone"

// SystemOneRoute describes an active routing plan, not backend health or
// empirically established accuracy. Reading it performs no model probes.
type SystemOneRoute struct {
	Model              string   `json:"model"`
	Recipe             string   `json:"recipe"`
	Algorithms         []string `json:"algorithms"`
	QuestionTypes      []string `json:"question_types"`
	ExecutionTimeoutMS int64    `json:"execution_timeout_ms"`
}

type SystemOneRoutes struct {
	Available bool             `json:"available"`
	Routes    []SystemOneRoute `json:"routes"`
}

type SystemOneRouteDiagnosticRequest struct {
	Model   string          `json:"model"`
	Request json.RawMessage `json:"request"`
}

func (SystemOneRouteDiagnosticRequest) JSONWire() any {
	return struct {
		Model   string              `json:"model"`
		Request api.DecisionRequest `json:"request"`
	}{}
}

func apiSystemOneRoutingRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(EndpointMetadata{Path: systemOneRoutingDiagnosticPath, Method: http.MethodGet, Description: "List active native recipe entrypoints for operator diagnostics without probing models; availability describes the routing plan, not backend health"}, routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig}, (*ClassificationAPIServer).handleSystemOneRoutes, jsonResponse[SystemOneRoutes](http.StatusOK, "Active native routing plans")),
		managedRoute(EndpointMetadata{Path: systemOneRoutingDiagnosticPath, Method: http.MethodPost, Description: "Run a native recipe using operator classify.invoke permission; the selected algorithm owns its execution deadline and physical call budget, while signals use their own timeouts; independent of public listener grants"}, routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational}, (*ClassificationAPIServer).handleSystemOneRouteDiagnostic, strictJSONBodyFor[SystemOneRouteDiagnosticRequest](), jsonResponse[map[string]any](http.StatusOK, "Selected native response and routing outcome"), errorResponses(400, 404, 413, 502, 503, 504)),
	}
}

func (s *ClassificationAPIServer) handleSystemOneRoutes(w http.ResponseWriter, _ *http.Request) {
	response := SystemOneRoutes{Routes: []SystemOneRoute{}}
	snapshot, router, release, ok := s.runtimeRegistry.AcquireSystemOne()
	if !ok {
		s.writeJSONResponse(w, http.StatusOK, response)
		return
	}
	defer release()
	cfg := snapshot.Config()
	response.Available = router != nil && cfg.RoutingEnabled()
	if response.Available {
		reachable := map[config.RecipeName]bool{}
		for _, recipe := range cfg.ReachableRoutingRecipes() {
			reachable[recipe.Name] = true
		}
		for _, entrypoint := range cfg.EffectiveEntrypoints(config.SystemOneAPI) {
			recipe, exists := cfg.RecipeByName(entrypoint.Recipe)
			if !exists || !reachable[entrypoint.Recipe] {
				continue
			}
			algorithms := []string{}
			var executionTimeout time.Duration
			for _, decision := range recipe.Profile.Decisions {
				if !decision.Algorithm.IsNative() || decision.Algorithm.Budget == nil {
					continue
				}
				if !slices.Contains(algorithms, decision.Algorithm.Type) {
					algorithms = append(algorithms, decision.Algorithm.Type)
				}
				duration, _ := time.ParseDuration(decision.Algorithm.Budget.Deadline)
				executionTimeout = max(executionTimeout, duration)
			}
			for _, model := range entrypoint.ModelNames {
				response.Routes = append(response.Routes, SystemOneRoute{Model: model, Recipe: string(recipe.Name), Algorithms: algorithms, QuestionTypes: []string{"choice", "score", "noul"}, ExecutionTimeoutMS: executionTimeout.Milliseconds()})
			}
		}
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleSystemOneRouteDiagnostic(w http.ResponseWriter, r *http.Request) {
	body, err := readJSONRequestBody(r, systemOneRequestLimit)
	if errors.Is(err, errRequestBodyTooLarge) {
		writeSystemOneError(w, http.StatusRequestEntityTooLarge, "request_too_large", "The request exceeds 2 MiB")
		return
	}
	var input SystemOneRouteDiagnosticRequest
	if err != nil || decodeStrictJSONBody(body, &input) != nil || input.Model == "" {
		writeSystemOneError(w, http.StatusBadRequest, "invalid_request", "Provide a native entrypoint and request")
		return
	}
	var task map[string]json.RawMessage
	if json.Unmarshal(input.Request, &task) != nil || task == nil {
		writeSystemOneError(w, http.StatusBadRequest, "invalid_request", "Provide a native request object")
		return
	}
	snapshot, router, release, ok := s.runtimeRegistry.AcquireSystemOne()
	if !ok || router == nil {
		if ok {
			release()
		}
		writeSystemOneError(w, http.StatusServiceUnavailable, "not_ready", "Native routing is not available")
		return
	}
	defer release()
	cfg := snapshot.Config()
	if !cfg.RoutingEnabled() {
		writeSystemOneError(w, http.StatusServiceUnavailable, "not_ready", "Native routing is not available")
		return
	}
	entrypoint, exists := cfg.ResolveEntrypoint(config.SystemOneAPI, input.Model)
	reachable := false
	for _, recipe := range cfg.ReachableRoutingRecipes() {
		reachable = reachable || recipe.Name == entrypoint.Recipe
	}
	if !exists || !reachable {
		writeSystemOneError(w, http.StatusNotFound, "model_not_found", "The native entrypoint is not active")
		return
	}
	task["model"], _ = json.Marshal(input.Model)
	nativeBody, _ := json.Marshal(task)
	nativeRequest, _ := http.NewRequestWithContext(r.Context(), http.MethodPost, "/v1/systemone", bytes.NewReader(nativeBody))
	// This managed route has already enforced classify.invoke. Its operator
	// capability never becomes an API key or changes a public listener grant.
	listener := &config.Listener{SystemOne: &config.ListenerSystemOne{Models: []string{input.Model}}}
	systemone.Handler(cfg, listener, retainedSystemOneInvoker(snapshot, router, ""))(w, nativeRequest)
}
