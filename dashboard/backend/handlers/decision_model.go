package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	runtimeapi "github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

const (
	decisionModelDiagnosticPath = "/api/v1/diagnostics/models/systemone"
	decisionModelRequestLimit   = 2 << 20
	decisionModelResponseLimit  = 4 << 20
)

type decisionModelDeployment struct {
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

type decisionModelCapabilities struct {
	ServingMode string                    `json:"serving_mode"`
	TimeoutMS   int                       `json:"timeout_ms"`
	Deployments []decisionModelDeployment `json:"deployments"`
}

type decisionModelTransport struct {
	upstream string
	provider routerauth.CredentialProvider
	client   *http.Client
}

// DecisionModelHandler only calls the configured Router/Engine. A request names
// a published deployment or served model ID, never a target URL or socket path.
func DecisionModelHandler(upstream string, providers ...routerauth.CredentialProvider) http.HandlerFunc {
	transport := decisionModelTransport{upstream: strings.TrimRight(upstream, "/"), client: &http.Client{Timeout: 35 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}
	if len(providers) > 0 {
		transport.provider = providers[0]
	}
	return transport.serveHTTP
}

func (t decisionModelTransport) serveHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	get := r.URL.Path == "/api/decision-model/capabilities" && r.Method == http.MethodGet
	post := r.URL.Path == "/api/decision-model/test" && r.Method == http.MethodPost
	if !get && !post {
		decisionModelError(w, 405, "method_not_allowed", "Unsupported Decision Model operation")
		return
	}
	var request struct {
		Deployment string          `json:"deployment"`
		Request    json.RawMessage `json:"request"`
	}
	if post {
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, decisionModelRequestLimit))
		decoder.DisallowUnknownFields()
		err := decoder.Decode(&request)
		var limit *http.MaxBytesError
		if errors.As(err, &limit) {
			decisionModelError(w, 413, "request_too_large", "The request exceeds 2 MiB")
			return
		}
		if err != nil || strings.TrimSpace(request.Deployment) == "" || len(request.Request) == 0 {
			decisionModelError(w, 400, "invalid_request", "Provide a deployment and a native System One request")
			return
		}
		var extra any
		if decoder.Decode(&extra) != io.EOF {
			decisionModelError(w, 400, "invalid_request", "Provide one JSON request")
			return
		}
	}
	ctx, cancel := context.WithTimeout(r.Context(), 35*time.Second)
	defer cancel()
	discoveryCtx, discoveryCancel := context.WithTimeout(ctx, 5*time.Second)
	capabilities, err := t.capabilities(discoveryCtx)
	discoveryCancel()
	if err != nil {
		decisionModelTransportError(w, err)
		return
	}
	if get {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(capabilities)
		return
	}
	var selected *decisionModelDeployment
	for i := range capabilities.Deployments {
		if capabilities.Deployments[i].ID == request.Deployment {
			selected = &capabilities.Deployments[i]
			break
		}
	}
	if selected == nil {
		decisionModelError(w, 404, "model_not_found", "The selected deployment is no longer available")
		return
	}
	if !selected.Ready {
		decisionModelError(w, 503, "not_ready", "The selected model is not ready")
		return
	}
	if len(selected.QuestionTypes) == 0 {
		decisionModelError(w, 422, "unsupported_surface", "The selected model does not serve System One questions")
		return
	}
	path := decisionModelDiagnosticPath
	body, err := json.Marshal(request)
	if capabilities.ServingMode == servingModeEngine {
		path = "/v1/systemone"
		var native map[string]json.RawMessage
		if json.Unmarshal(request.Request, &native) != nil || native == nil {
			decisionModelError(w, 400, "invalid_request", "The native request must be a JSON object")
			return
		}
		native["model"], _ = json.Marshal(selected.Model)
		body, err = json.Marshal(native)
	}
	if err != nil {
		decisionModelError(w, 400, "invalid_request", "Unable to encode request")
		return
	}
	executionCtx, executionCancel := context.WithTimeout(ctx, 30*time.Second)
	defer executionCancel()
	var response *decisionModelResponse
	var data []byte
	if capabilities.ServingMode == servingModeEngine && instanceEngineActive(executionCtx) {
		body, err = json.Marshal(request)
		if err == nil {
			var code int
			code, data, err = instanceRequest(executionCtx, http.MethodPost, "/systemone", body)
			response = &decisionModelResponse{StatusCode: code, Header: make(http.Header)}
		}
	} else {
		response, data, err = t.request(executionCtx, http.MethodPost, path, body)
	}
	if err != nil {
		decisionModelTransportError(w, err)
		return
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 && response.StatusCode < 400 || !json.Valid(data) {
		decisionModelError(w, 502, "invalid_response", "The model runtime returned an invalid response")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if timing := response.Header.Get("Server-Timing"); timing != "" {
		w.Header().Set("Server-Timing", timing)
	}
	w.WriteHeader(response.StatusCode)
	_, _ = w.Write(data)
}

type decisionModelResponse struct {
	StatusCode int
	Header     http.Header
}

func (t decisionModelTransport) request(ctx context.Context, method, path string, body []byte) (*decisionModelResponse, []byte, error) {
	request, err := http.NewRequestWithContext(ctx, method, t.upstream+path, bytes.NewReader(body))
	if err != nil {
		return nil, nil, err
	}
	request.Header.Set("Accept", "application/json")
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if err = routerauth.RewriteAuthorization(request, t.provider); err != nil {
		return nil, nil, err
	}
	response, err := t.client.Do(request)
	if err != nil {
		return nil, nil, err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, decisionModelResponseLimit+1))
	if err != nil {
		return nil, nil, err
	}
	if len(data) > decisionModelResponseLimit {
		return nil, nil, errors.New("response limit exceeded")
	}
	return &decisionModelResponse{StatusCode: response.StatusCode, Header: response.Header}, data, nil
}

func (t decisionModelTransport) capabilities(ctx context.Context) (decisionModelCapabilities, error) {
	result := decisionModelCapabilities{ServingMode: servingModeRouter, TimeoutMS: 30000, Deployments: []decisionModelDeployment{}}
	if instanceEngineActive(ctx) {
		code, body, err := instanceRequest(ctx, http.MethodGet, "/models", nil)
		if err != nil {
			return result, err
		}
		return engineDecisionCapabilities(code, body)
	}
	response, body, err := t.request(ctx, http.MethodGet, decisionModelDiagnosticPath, nil)
	if err != nil {
		return result, err
	}
	if response.StatusCode == 200 {
		if err = json.Unmarshal(body, &result); err != nil {
			return result, err
		}
		return result, nil
	}
	if response.StatusCode != 404 {
		return result, errors.New("runtime capability discovery failed")
	}
	// A missing Router route is not proof of Engine mode. Confirm the versioned
	// Model Runtime identity before calling its native model and inference APIs.
	if !t.isEngine(ctx) {
		if ctx.Err() != nil {
			return result, ctx.Err()
		}
		return result, errDecisionModelUnsupported
	}
	response, body, err = t.request(ctx, http.MethodGet, "/v1/models", nil)
	if err != nil {
		return result, err
	}
	return engineDecisionCapabilities(response.StatusCode, body)
}

func engineDecisionCapabilities(code int, body []byte) (decisionModelCapabilities, error) {
	result := decisionModelCapabilities{ServingMode: servingModeEngine, TimeoutMS: 30000, Deployments: []decisionModelDeployment{}}
	var models runtimeapi.ModelList
	if code != 200 || json.Unmarshal(body, &models) != nil {
		return result, errors.New("model inventory unavailable")
	}
	result.ServingMode = servingModeEngine
	for _, card := range models.Data {
		item := decisionModelDeployment{ID: card.Id, Model: card.Id, Ready: card.Ready, Family: card.Family, Surfaces: card.Surfaces, QuestionTypes: []string{}}
		if card.Repo != nil {
			item.Repo = *card.Repo
		}
		decisions := false
		for _, surface := range card.Surfaces {
			if surface == "decisions" {
				decisions = true
			}
		}
		if decisions {
			item.QuestionTypes = []string{"choice", "score", "noul"}
			if card.QuestionTypes != nil && len(*card.QuestionTypes) > 0 {
				item.QuestionTypes = *card.QuestionTypes
			}
		} else {
			item.UnavailableReason = "This model does not serve System One questions"
		}
		if !card.Ready {
			item.UnavailableReason = "Model is not ready"
		}
		if card.Presets != nil {
			item.Presets = *card.Presets
		}
		if card.Limits != nil {
			if card.Limits.MaxInputTokens != nil {
				item.MaxInputTokens = *card.Limits.MaxInputTokens
			}
			if card.Limits.MaxScanTokens != nil {
				item.MaxScanTokens = *card.Limits.MaxScanTokens
			}
		}
		result.Deployments = append(result.Deployments, item)
	}
	return result, nil
}

func (t decisionModelTransport) isEngine(ctx context.Context) bool {
	response, body, err := t.request(ctx, http.MethodGet, "/health", nil)
	if err != nil || response.StatusCode != 200 && response.StatusCode != 503 {
		return false
	}
	var health struct {
		APIVersion string `json:"api_version"`
		Status     string `json:"status"`
	}
	if json.Unmarshal(body, &health) != nil || !engineAPIVersion.MatchString(health.APIVersion) {
		return false
	}
	response, body, err = t.request(ctx, http.MethodGet, "/openapi.yaml", nil)
	if err != nil || response.StatusCode != 200 {
		return false
	}
	var contract struct {
		OpenAPI string `yaml:"openapi"`
		Info    struct {
			Title   string `yaml:"title"`
			Version string `yaml:"version"`
		} `yaml:"info"`
		Paths map[string]map[string]any `yaml:"paths"`
	}
	if yaml.Unmarshal(body, &contract) != nil || !strings.HasPrefix(contract.OpenAPI, "3.") || contract.Info.Title != "vLLM Semantic Router Model Runtime" || contract.Info.Version != health.APIVersion {
		return false
	}
	for path, method := range map[string]string{"/v1/systemone": "post", "/v1/models": "get"} {
		if _, ok := contract.Paths[path][method]; !ok {
			return false
		}
	}
	return true
}

var errDecisionModelUnsupported = errors.New("system one diagnostic API unavailable")

func decisionModelTransportError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, errDecisionModelUnsupported):
		decisionModelError(w, 503, "unsupported_runtime", "Update the Router to enable System One testing, or connect a compatible Model Engine")
	case errors.Is(err, context.DeadlineExceeded):
		decisionModelError(w, 504, "deadline_exceeded", "The System One request exceeded its deadline")
	case errors.Is(err, context.Canceled):
		decisionModelError(w, 503, "canceled", "The System One request was canceled")
	default:
		decisionModelError(w, 502, "runtime_unavailable", "Unable to reach the configured model runtime")
	}
}

func decisionModelError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"code": code, "message": message}})
}
