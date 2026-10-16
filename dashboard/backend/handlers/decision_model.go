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

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
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

// DecisionModelHandler only calls the configured frontend. A request names
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
	body, err := json.Marshal(request)
	if err != nil {
		decisionModelError(w, 400, "invalid_request", "Unable to encode request")
		return
	}
	executionCtx, executionCancel := context.WithTimeout(ctx, 30*time.Second)
	defer executionCancel()
	response, data, err := t.request(executionCtx, http.MethodPost, decisionModelDiagnosticPath, body)
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
	return result, errDecisionModelUnsupported
}

var errDecisionModelUnsupported = errors.New("system one diagnostic API unavailable")

func decisionModelTransportError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, errDecisionModelUnsupported):
		decisionModelError(w, 503, "unsupported_runtime", "The configured frontend does not expose System One testing")
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
