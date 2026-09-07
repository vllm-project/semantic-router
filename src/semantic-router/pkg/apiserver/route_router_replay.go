//go:build !windows && cgo

package apiserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/redaction"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func apiRouterReplayRoutes() []apiRoute {
	policy := routePolicy{Permission: PermReplayRead, Sensitivity: SensitivityReplay}
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: "/v1/router_replay", Method: "GET", Description: "List Router Replay records"},
			policy,
			(*ClassificationAPIServer).handleRouterReplay,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/router_replay/", Method: "GET", Description: "List Router Replay records (trailing-slash compatibility)"},
			policy,
			(*ClassificationAPIServer).handleRouterReplay,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/router_replay/aggregate", Method: "GET", Description: "Aggregate Router Replay routing and cost metadata"},
			policy,
			(*ClassificationAPIServer).handleRouterReplay,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/router_replay/trajectory", Method: "GET", Description: "Build a Router Replay session trajectory"},
			policy,
			(*ClassificationAPIServer).handleRouterReplay,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/router_replay/{id}", Method: "GET", Description: "Read one Router Replay record"},
			policy,
			(*ClassificationAPIServer).handleRouterReplay,
		),
	}
}

func (s *ClassificationAPIServer) handleRouterReplay(w http.ResponseWriter, r *http.Request) {
	runtime, release := s.currentReplayRuntime()
	defer release()
	if runtime == nil {
		s.writeJSONResponse(w, http.StatusNotFound, replayManagementError(
			http.StatusNotFound,
			"router replay is disabled or unavailable",
		))
		return
	}

	requestTarget := r.URL.Path
	if r.URL.RawQuery != "" {
		requestTarget += "?" + r.URL.RawQuery
	}
	response, handled := runtime.HandleReplayRequest(r.Method, requestTarget)
	if !handled {
		s.writeJSONResponse(w, http.StatusNotFound, replayManagementError(
			http.StatusNotFound,
			"router replay is disabled or unavailable",
		))
		return
	}

	body := response.Body
	if !s.canViewReplayDetails(r) {
		redacted, _, err := redaction.RedactResponseBody(body)
		if err != nil {
			s.writeJSONResponse(w, http.StatusInternalServerError, replayManagementError(
				http.StatusInternalServerError,
				"router replay response could not be safely redacted",
			))
			return
		}
		body = redacted
	}

	statusCode := response.StatusCode
	if statusCode < 100 || statusCode > 599 {
		statusCode = http.StatusInternalServerError
		body = []byte(`{"error":{"message":"Internal server error","type":"internal_error","code":500}}`)
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

func replayManagementError(statusCode int, message string) map[string]interface{} {
	return map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    "invalid_request_error",
			"code":    statusCode,
		},
	}
}
