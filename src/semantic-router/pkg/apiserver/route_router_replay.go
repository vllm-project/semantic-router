//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/redaction"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func apiRouterReplayRoutes() []apiRoute {
	policy := routePolicy{Permission: PermReplayRead, Sensitivity: SensitivityReplay}
	return []apiRoute{
		managedRoute(
			EndpointMetadata{
				Path:        apiObservabilityReplaysPath,
				Method:      "GET",
				Description: "List Router Replay records",
				Parameters:  routerReplayListParameters(),
			},
			policy,
			(*ClassificationAPIServer).handleRouterReplay,
			pluginOperationFor(config.DecisionPluginRouterReplay, "read"),
			pluginOperationFor(config.DecisionPluginShadowDispatch, "read"),
			jsonResponse[routerreplay.ListResponse](http.StatusOK, "Successful replay response"),
			errorResponses(http.StatusBadRequest, http.StatusNotFound, http.StatusInternalServerError),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiObservabilityReplaysPath + "/aggregate",
				Method:      "GET",
				Description: "Aggregate Router Replay routing and cost metadata",
				Parameters:  routerReplayFilterParameters(),
			},
			policy,
			(*ClassificationAPIServer).handleRouterReplay,
			pluginOperationFor(config.DecisionPluginRouterReplay, "read"),
			pluginOperationFor(config.DecisionPluginShadowDispatch, "read"),
			jsonResponse[routerreplay.AggregateResponse](http.StatusOK, "Successful replay response"),
			errorResponses(http.StatusBadRequest, http.StatusNotFound, http.StatusInternalServerError),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiObservabilityReplaysPath + "/trajectory",
				Method:      "GET",
				Description: "Build a recipe-scoped session trajectory with each recorded routing result",
				Parameters: []OpenAPIParameter{
					requiredQueryParameter("session_id", "Logical session whose tool trajectory should be returned.", "string"),
					queryParameter("recipe", "Exact recipe scope; an empty value selects legacy unscoped records. Required when the session ID occurs in multiple recipes.", "string"),
				},
			},
			policy,
			(*ClassificationAPIServer).handleRouterReplay,
			pluginOperationFor(config.DecisionPluginRouterReplay, "read"),
			pluginOperationFor(config.DecisionPluginShadowDispatch, "read"),
			jsonResponse[routerreplay.TrajectoryResponse](http.StatusOK, "Successful replay response"),
			errorResponses(http.StatusBadRequest, http.StatusNotFound, http.StatusInternalServerError),
		),
		managedRoute(
			EndpointMetadata{Path: apiObservabilityReplaysPath + "/{id}", Method: "GET", Description: "Read one Router Replay record"},
			policy,
			(*ClassificationAPIServer).handleRouterReplay,
			pluginOperationFor(config.DecisionPluginRouterReplay, "read"),
			pluginOperationFor(config.DecisionPluginShadowDispatch, "read"),
			jsonResponse[routerreplay.RoutingRecord](http.StatusOK, "Successful replay response"),
			errorResponses(http.StatusBadRequest, http.StatusNotFound, http.StatusInternalServerError),
		),
	}
}

func routerReplayFilterParameters() []OpenAPIParameter {
	return []OpenAPIParameter{
		queryParameter("search", "Case-insensitive text search across replay records.", "string"),
		queryParameter("recipe", "Filter by recipe name.", "string"),
		queryParameter("decision", "Filter by decision name.", "string"),
		queryParameter("model", "Filter by selected model.", "string"),
		queryParameter("session_id", "Filter by logical session identifier.", "string"),
		queryParameter("cache_status", "Filter by response-cache outcome.", "string", "all", "cached", "streamed"),
	}
}

func routerReplayListParameters() []OpenAPIParameter {
	parameters := routerReplayFilterParameters()
	return append(parameters,
		queryParameter("limit", "Maximum records; defaults to 20 and is capped at 100.", "integer"),
		queryParameter("offset", "Zero-based result offset.", "integer"),
		queryParameter("showDetails", "Include full replay details when the principal has replay_detail permission.", "boolean"),
	)
}

func (s *ClassificationAPIServer) handleRouterReplay(w http.ResponseWriter, r *http.Request) {
	runtime, release := s.currentReplayRuntime()
	defer release()
	if runtime == nil {
		s.writeErrorResponse(w, http.StatusNotFound, "REPLAY_UNAVAILABLE", "router replay is disabled or unavailable")
		return
	}

	requestTarget := r.URL.Path
	if r.URL.RawQuery != "" {
		requestTarget += "?" + r.URL.RawQuery
	}
	response, handled := runtime.HandleReplayRequest(r.Method, requestTarget)
	if !handled {
		s.writeErrorResponse(w, http.StatusNotFound, "REPLAY_UNAVAILABLE", "router replay is disabled or unavailable")
		return
	}

	body := response.Body
	if !s.canViewReplayDetails(r) {
		redacted, _, err := redaction.RedactResponseBody(body)
		if err != nil {
			s.writeErrorResponse(w, http.StatusInternalServerError, "REPLAY_REDACTION_ERROR", "router replay response could not be safely redacted")
			return
		}
		body = redacted
	}

	statusCode := response.StatusCode
	if statusCode < 100 || statusCode > 599 {
		s.writeErrorResponse(w, http.StatusInternalServerError, "REPLAY_INVALID_STATUS", "Internal server error")
		return
	}
	if statusCode >= 400 {
		var payload struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		message := http.StatusText(statusCode)
		if json.Unmarshal(body, &payload) == nil && payload.Error.Message != "" {
			message = payload.Error.Message
		}
		s.writeErrorResponse(w, statusCode, fmt.Sprintf("REPLAY_HTTP_%d", statusCode), message)
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(statusCode)
	_, _ = w.Write(body)
}

func (s *ClassificationAPIServer) currentReplayRuntime() (routerruntime.ReplayRuntime, func()) {
	if s == nil || s.runtimeRegistry == nil {
		return nil, func() {}
	}
	return s.runtimeRegistry.AcquireReplayRuntime()
}

func (s *ClassificationAPIServer) canViewReplayDetails(r *http.Request) bool {
	if r == nil {
		return false
	}
	principal, ok := managementPrincipalFromContext(r.Context())
	if !ok {
		return false
	}
	roles := s.managementAuthPolicy().Roles
	return principal.hasPermission(PermReplayDetail, roles)
}
