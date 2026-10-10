//go:build !windows

package apiserver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

const (
	systemOneDiagnosticPath = apiDiagnosticsPath + "/models/systemone"
	systemOneRequestLimit   = 2 << 20
)

// SystemOneDiagnosticRequest selects a published deployment, never an endpoint.
// The request is the runtime's native /v1/systemone document; the deployment
// determines its model. Nested raw JSON retains question and criteria order.
type SystemOneDiagnosticRequest struct {
	Deployment       string          `json:"deployment"`
	ExpectedArtifact string          `json:"expected_artifact,omitempty"`
	Request          json.RawMessage `json:"request"`
}

// JSONWire documents the native request using its generated canonical contract.
func (SystemOneDiagnosticRequest) JSONWire() any {
	return struct {
		Deployment       string              `json:"deployment"`
		ExpectedArtifact string              `json:"expected_artifact,omitempty"`
		Request          api.DecisionRequest `json:"request"`
	}{}
}

type SystemOneDeployment struct {
	Artifact          string   `json:"artifact,omitempty"`
	ID                string   `json:"id"`
	Model             string   `json:"model"`
	Ready             bool     `json:"ready"`
	QuestionTypes     []string `json:"question_types"`
	Surfaces          []string `json:"surfaces"`
	UnavailableReason string   `json:"unavailable_reason,omitempty"`
	Repo              string   `json:"repo,omitempty"`
	Family            string   `json:"family,omitempty"`
	Presets           []string `json:"presets,omitempty"`
	MaxInputTokens    int      `json:"max_input_tokens,omitempty"`
	MaxScanTokens     int      `json:"max_scan_tokens,omitempty"`
}

type SystemOneCapabilities struct {
	Deployments []SystemOneDeployment `json:"deployments"`
}

func apiSystemOneRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(EndpointMetadata{Path: systemone.ForwardPath, Method: http.MethodPost, Description: "Forward a public native inference or discovery request through one active listener grant; requires management authorization and the original listener credentials"}, routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational}, (*ClassificationAPIServer).handleSystemOneForward, strictJSONBodyWithLimitFor[systemone.ForwardRequest](systemOneForwardRequestLimit), jsonResponse[map[string]any](http.StatusOK, "Native response or listener model discovery"), errorResponses(400, 401, 403, 404, 409, 413, 502, 503, 504)),
		managedRoute(EndpointMetadata{Path: apiRootPath + "/instance", Method: http.MethodGet, Description: "Read the serving frontend capability mode and default native deployment"}, routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig}, (*ClassificationAPIServer).handleInstanceStatus, jsonResponse[InstanceStatus](http.StatusOK, "Active frontend state")),
		managedRoute(EndpointMetadata{Path: apiDiagnosticsPath + "/models/tasks", Method: http.MethodGet, Description: "List shared judgment task templates, structural model capabilities and binding provenance"}, routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig}, (*ClassificationAPIServer).handleDecisionTasks, jsonResponse[modelservice.TaskCatalogResponse](200, "Task templates, capabilities and effective bindings")),
		managedRoute(EndpointMetadata{Path: systemOneDiagnosticPath, Method: http.MethodGet, Description: "List published model deployments and their native System One question capabilities"}, routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig}, (*ClassificationAPIServer).handleSystemOneCapabilities, jsonResponse[SystemOneCapabilities](200, "Published deployment capabilities")),
		managedRoute(EndpointMetadata{Path: systemOneDiagnosticPath, Method: http.MethodPost, Description: "Test native System One questions against a published deployment; preserves choice, score, noul, set, span, usage and metadata; 2 MiB request, 4 MiB response, 30 second deadline"}, routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational}, (*ClassificationAPIServer).handleSystemOneDiagnostic, strictJSONBodyFor[SystemOneDiagnosticRequest](), jsonResponse[api.DecisionResponse](200, "Native System One response; per-question errors are preserved"), errorResponses(400, 404, 409, 413, 422, 429, 500, 502, 503, 504)),
	}
}

func (s *ClassificationAPIServer) handleSystemOneCapabilities(w http.ResponseWriter, _ *http.Request) {
	response := SystemOneCapabilities{Deployments: []SystemOneDeployment{}}
	if manager := modelservice.DefaultManager(); manager != nil {
		for _, status := range manager.Statuses() {
			item := SystemOneDeployment{ID: status.Name, Model: status.Model, Artifact: status.Artifact, Ready: status.Ready, QuestionTypes: []string{}, Surfaces: []string{}}
			if !status.Ready {
				item.UnavailableReason = "Deployment is not ready"
			}
			if card := status.Card; card != nil {
				item.Surfaces = card.Surfaces
				item.Repo, item.Family, item.Presets = card.Repo, card.Family, card.Presets
				item.MaxInputTokens, item.MaxScanTokens = card.MaxInputTokens, card.MaxScanTokens
				if card.Serves("decisions") {
					for _, kind := range []string{"choice", "score", "noul", "span", "set"} {
						if card.Answers(kind) {
							item.QuestionTypes = append(item.QuestionTypes, kind)
						}
					}
				} else {
					item.UnavailableReason = "This model does not serve System One questions"
				}
			}
			response.Deployments = append(response.Deployments, item)
		}
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleSystemOneDiagnostic(w http.ResponseWriter, r *http.Request) {
	body, err := readJSONRequestBody(r, systemOneRequestLimit)
	if err != nil {
		code, status := "invalid_request", 400
		if errors.Is(err, errRequestBodyTooLarge) {
			code, status = "request_too_large", 413
		}
		writeSystemOneError(w, status, code, "Unable to read bounded System One request")
		return
	}
	var request SystemOneDiagnosticRequest
	if decodeStrictJSONBody(body, &request) != nil || strings.TrimSpace(request.Deployment) == "" || len(request.Request) == 0 {
		writeSystemOneError(w, 400, "invalid_request", "Provide a deployment and a native System One request")
		return
	}
	manager := modelservice.DefaultManager()
	if manager == nil {
		writeSystemOneError(w, 503, "not_ready", "The model runtime is not ready")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	result, err := runRetainedAPIWork(ctx, func() {}, func(ctx context.Context) (modelservice.SystemOneResult, error) {
		return manager.SystemOneForArtifact(ctx, request.Deployment, request.ExpectedArtifact, request.Request)
	})
	if len(result.Body) > 0 {
		w.Header().Set("Content-Type", "application/json")
		if result.ServerTiming != "" {
			w.Header().Set("Server-Timing", result.ServerTiming)
		}
		w.WriteHeader(result.Status)
		_, _ = w.Write(result.Body)
		return
	}
	switch {
	case errors.Is(err, modelservice.ErrSystemOneArtifactChanged):
		writeSystemOneError(w, 409, "model_deployment_pending", "The published model requires deployment")
	case errors.Is(err, modelservice.ErrUnknownDeployment):
		writeSystemOneError(w, 404, "model_not_found", "The selected deployment is not published")
	case errors.Is(err, modelservice.ErrRejected):
		writeSystemOneError(w, 400, "invalid_request", "The native request must be a JSON object")
	case errors.Is(err, context.DeadlineExceeded):
		writeSystemOneError(w, 504, "deadline_exceeded", "System One test exceeded its deadline")
	case errors.Is(err, context.Canceled):
		writeSystemOneError(w, 503, "canceled", "System One test was canceled")
	case errors.Is(err, modelservice.ErrUnavailable):
		writeSystemOneError(w, 503, "not_ready", "The selected deployment is not ready")
	default:
		writeSystemOneError(w, 502, "runtime_unavailable", "The model runtime did not return a valid System One response")
	}
}

func writeSystemOneError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"code": code, "message": message}})
}

// InstanceStatus describes the active snapshot, never a pending saved document.
type InstanceStatus struct {
	ObservedMode     string `json:"observed_mode"`
	ActiveDeployment string `json:"active_deployment,omitempty"`
	Model            string `json:"model,omitempty"`
}

func (s *ClassificationAPIServer) handleInstanceStatus(w http.ResponseWriter, _ *http.Request) {
	state := InstanceStatus{ObservedMode: "unknown"}
	if snapshot := s.activeConfigSnapshot(); snapshot != nil {
		cfg := snapshot.Config()
		state.ObservedMode = "engine"
		if cfg.RoutingEnabled() {
			state.ObservedMode = "router"
		}
		name, deployment, ok, err := cfg.DecisionModelDeployment()
		if err == nil && ok {
			state.ActiveDeployment, state.Model = name, deployment.Artifact
		}
	}
	s.writeJSONResponse(w, http.StatusOK, state)
}
