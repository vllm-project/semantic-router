package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/dashboard/backend/accesscontrol"
	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

const (
	maxGatewayBody       = 32 << 20
	gatewayAccountingTTL = 5 * time.Second
)

type AccessGatewayHandler struct {
	service  *accesscontrol.Service
	upstream *url.URL
	client   *http.Client
}

type chatCompletionEnvelope struct {
	Model     string          `json:"model"`
	MaxTokens int64           `json:"max_tokens"`
	MaxOutput int64           `json:"max_completion_tokens"`
	Messages  json.RawMessage `json:"messages"`
	Stream    bool            `json:"stream"`
}

func NewAccessGatewayHandler(service *accesscontrol.Service, upstream string) (*AccessGatewayHandler, error) {
	parsed, err := url.Parse(strings.TrimSpace(upstream))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, errors.New("TARGET_ENVOY_URL must be a valid absolute URL when access control is enabled")
	}
	return &AccessGatewayHandler{
		service: service, upstream: parsed,
		client: &http.Client{Transport: http.DefaultTransport, Timeout: 0},
	}, nil
}

func (h *AccessGatewayHandler) Models(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	principal, ok := h.authenticate(w, r)
	if !ok {
		return
	}
	h.models(w, r, principal)
}

func (h *AccessGatewayHandler) DashboardModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	principal, ok := h.authenticateDashboard(w, r)
	if !ok {
		return
	}
	h.models(w, r, principal)
}

func (h *AccessGatewayHandler) models(w http.ResponseWriter, r *http.Request, principal *accesscontrol.Principal) {
	target := h.resolve("/v1/models")
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, target, nil)
	if err != nil {
		writeAccessError(w, http.StatusBadGateway, "could not create upstream request")
		return
	}
	response, err := h.client.Do(req)
	if err != nil {
		writeAccessError(w, http.StatusBadGateway, "model catalog upstream is unavailable")
		return
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, maxGatewayBody))
	if err != nil {
		writeAccessError(w, http.StatusBadGateway, "could not read upstream model catalog")
		return
	}
	if response.StatusCode >= 400 {
		copyGatewayResponse(w, response, body)
		return
	}
	var catalog struct {
		Object string           `json:"object"`
		Data   []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(body, &catalog); err != nil {
		writeAccessError(w, http.StatusBadGateway, "upstream returned an invalid model catalog")
		return
	}
	filtered := catalog.Data[:0]
	for _, model := range catalog.Data {
		id, _ := model["id"].(string)
		if accesscontrol.ModelAllowed(id, principal.ModelPatterns) {
			filtered = append(filtered, model)
		}
	}
	catalog.Data = filtered
	setJSONContentType(w)
	_ = json.NewEncoder(w).Encode(catalog)
}

func (h *AccessGatewayHandler) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	principal, ok := h.authenticate(w, r)
	if !ok {
		return
	}
	h.chatCompletions(w, r, principal, "api_key")
}

func (h *AccessGatewayHandler) DashboardChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	principal, ok := h.authenticateDashboard(w, r)
	if !ok {
		return
	}
	h.chatCompletions(w, r, principal, "dashboard_session")
}

func (h *AccessGatewayHandler) chatCompletions(w http.ResponseWriter, r *http.Request, principal *accesscontrol.Principal, credentialType string) {
	started := time.Now()
	requestID := uuid.NewString()
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxGatewayBody))
	if err != nil {
		writeAccessError(w, http.StatusRequestEntityTooLarge, "request body is too large")
		return
	}
	var envelope chatCompletionEnvelope
	if parseErr := json.Unmarshal(body, &envelope); parseErr != nil || strings.TrimSpace(envelope.Model) == "" {
		h.recordGatewayUsage(principal, requestID, envelope.Model, http.StatusBadRequest, started, 0, 0, 0, 0, "invalid_request", gatewayUsageMetadata(body, nil, envelope.Stream, credentialType))
		writeAccessError(w, http.StatusBadRequest, "model and a valid JSON body are required")
		return
	}
	if !accesscontrol.ModelAllowed(envelope.Model, principal.ModelPatterns) {
		h.recordGatewayUsage(principal, requestID, envelope.Model, http.StatusForbidden, started, 0, 0, 0, 0, "model_forbidden", gatewayUsageMetadata(body, nil, envelope.Stream, credentialType))
		writeAccessError(w, http.StatusForbidden, "this API key is not authorized for the requested model")
		return
	}
	if envelope.Stream {
		body, err = requestBodyWithStreamUsage(body)
		if err != nil {
			h.recordGatewayUsage(principal, requestID, envelope.Model, http.StatusBadRequest, started, 0, 0, 0, 0, "invalid_request", gatewayUsageMetadata(body, nil, envelope.Stream, credentialType))
			writeAccessError(w, http.StatusBadRequest, "a valid JSON body is required")
			return
		}
	}
	maxOutput := max(envelope.MaxTokens, envelope.MaxOutput)
	if maxOutput <= 0 {
		maxOutput = 1024
	}
	// Reserve at most one input token per JSON byte plus the requested output
	// ceiling. This is intentionally conservative: reconciliation refunds the
	// difference after usage arrives, while concurrent replicas cannot admit
	// requests based on an optimistic chars-per-token estimate.
	estimatedTokens := max(int64(len(envelope.Messages))+maxOutput, 1)
	reservation, err := h.service.Quota().Reserve(r.Context(), principal.Budgets, estimatedTokens)
	if err != nil {
		var quotaErr *accesscontrol.QuotaError
		if errors.As(err, &quotaErr) {
			h.recordGatewayUsage(principal, requestID, envelope.Model, http.StatusTooManyRequests, started, 0, 0, 0, 0, "quota_exceeded", gatewayUsageMetadata(body, nil, envelope.Stream, credentialType))
			w.Header().Set("Retry-After", "60")
			writeAccessError(w, http.StatusTooManyRequests, quotaErr.Error())
			return
		}
		h.recordGatewayUsage(principal, requestID, envelope.Model, http.StatusServiceUnavailable, started, 0, 0, 0, 0, "quota_unavailable", gatewayUsageMetadata(body, nil, envelope.Stream, credentialType))
		writeAccessError(w, http.StatusServiceUnavailable, "global quota service is unavailable")
		return
	}

	h.proxyChatCompletion(
		w, r, principal, credentialType, started, requestID, body, envelope, estimatedTokens, reservation,
	)
}

func (h *AccessGatewayHandler) proxyChatCompletion(
	w http.ResponseWriter,
	r *http.Request,
	principal *accesscontrol.Principal,
	credentialType string,
	started time.Time,
	requestID string,
	body []byte,
	envelope chatCompletionEnvelope,
	estimatedTokens int64,
	reservation *accesscontrol.Reservation,
) {
	target := h.resolve("/v1/chat/completions")
	upstreamRequest, err := http.NewRequestWithContext(r.Context(), http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		h.reconcileQuota(reservation, 0, requestID)
		h.recordGatewayUsage(principal, requestID, envelope.Model, http.StatusBadGateway, started, 0, 0, 0, 0, "gateway_request_error", gatewayUsageMetadata(body, nil, envelope.Stream, credentialType))
		writeAccessError(w, http.StatusBadGateway, "could not create upstream request")
		return
	}
	upstreamRequest.Header.Set("Content-Type", "application/json")
	upstreamRequest.Header.Set("Accept", r.Header.Get("Accept"))
	upstreamRequest.Header.Set("X-Request-ID", requestID)
	upstreamRequest.Header.Set("X-VLLM-SR-API-Key-ID", principal.Key.ID)
	if principal.Key.UserID != "" {
		upstreamRequest.Header.Set("X-VLLM-SR-User-ID", principal.Key.UserID)
	}
	if principal.Key.TeamID != "" {
		upstreamRequest.Header.Set("X-VLLM-SR-Team-ID", principal.Key.TeamID)
	}
	response, upstreamErr := h.client.Do(upstreamRequest)
	if upstreamErr != nil {
		h.reconcileQuota(reservation, 0, requestID)
		h.recordGatewayUsage(principal, requestID, envelope.Model, http.StatusBadGateway, started, 0, 0, 0, 0, "upstream_unavailable", gatewayUsageMetadata(body, nil, envelope.Stream, credentialType))
		writeAccessError(w, http.StatusBadGateway, "inference upstream is unavailable")
		return
	}
	defer response.Body.Close()
	for key, values := range response.Header {
		if isGatewayResponseHeader(key) {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
	}
	w.Header().Set("X-Request-ID", requestID)
	w.WriteHeader(response.StatusCode)

	capture := &limitedCapture{limit: maxGatewayBody}
	ttft := int64(0)
	var streamErr error
	buffer := make([]byte, 32*1024)
	for {
		read, readErr := response.Body.Read(buffer)
		if read > 0 {
			if ttft == 0 {
				ttft = time.Since(started).Milliseconds()
			}
			_, _ = capture.Write(buffer[:read])
			_, _ = w.Write(buffer[:read])
			if flusher, available := w.(http.Flusher); available && envelope.Stream {
				flusher.Flush()
			}
		}
		if readErr != nil {
			if !errors.Is(readErr, io.EOF) {
				streamErr = readErr
				log.Printf("access gateway response stream failed for request %s: %v", requestID, readErr)
			}
			break
		}
	}
	prompt, completion, total := parseGatewayUsage(capture.Bytes(), envelope.Stream)
	if total == 0 {
		total = prompt + completion
	}
	// A successful stream is not required to include the optional final usage
	// chunk. Keep the conservative reservation in that case instead of
	// refunding an unmeasured request and weakening the shared TPM limit.
	usageEstimated := total == 0 && response.StatusCode < http.StatusBadRequest
	if usageEstimated {
		total = estimatedTokens
	}
	h.reconcileQuota(reservation, total, requestID)
	errorCode := ""
	if streamErr != nil {
		errorCode = "upstream_stream_error"
	} else if response.StatusCode >= 400 {
		errorCode = "upstream_error"
	}
	metadata := gatewayUsageMetadata(body, capture.Bytes(), envelope.Stream, credentialType)
	if usageEstimated {
		metadata["usageEstimated"] = true
	}
	h.recordGatewayUsage(principal, requestID, envelope.Model, response.StatusCode, started, ttft, prompt, completion, total, errorCode, metadata)
}

func (h *AccessGatewayHandler) authenticate(w http.ResponseWriter, r *http.Request) (*accesscontrol.Principal, bool) {
	header := strings.TrimSpace(r.Header.Get("Authorization"))
	parts := strings.SplitN(header, " ", 2)
	if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") {
		writeAccessError(w, http.StatusUnauthorized, "a bearer API key is required")
		return nil, false
	}
	principal, err := h.service.Authenticate(r.Context(), strings.TrimSpace(parts[1]))
	if err != nil {
		writeAccessError(w, http.StatusUnauthorized, "invalid or inactive API key")
		return nil, false
	}
	return principal, true
}

func (h *AccessGatewayHandler) authenticateDashboard(w http.ResponseWriter, r *http.Request) (*accesscontrol.Principal, bool) {
	session, ok := auth.AuthFromContext(r)
	if !ok || strings.TrimSpace(session.UserID) == "" {
		writeAccessError(w, http.StatusUnauthorized, "a Dashboard session is required")
		return nil, false
	}
	principal, err := h.service.PrincipalForDashboardUser(r.Context(), session.AccessUserID)
	if err != nil {
		writeAccessError(w, http.StatusForbidden, "this Dashboard user has no active model access")
		return nil, false
	}
	return principal, true
}

func (h *AccessGatewayHandler) reconcileQuota(reservation *accesscontrol.Reservation, total int64, requestID string) {
	ctx, cancel := context.WithTimeout(context.Background(), gatewayAccountingTTL)
	defer cancel()
	if err := h.service.Quota().Reconcile(ctx, reservation, total); err != nil {
		log.Printf("access gateway quota reconciliation failed for request %s: %v", requestID, err)
	}
}

func (h *AccessGatewayHandler) recordGatewayUsage(principal *accesscontrol.Principal, requestID, model string, status int, started time.Time, ttft, prompt, completion, total int64, errorCode string, metadata map[string]any) {
	ctx, cancel := context.WithTimeout(context.Background(), gatewayAccountingTTL)
	defer cancel()
	if err := h.service.RecordUsage(ctx, accesscontrol.UsageEvent{
		RequestID: requestID, KeyID: principal.Key.ID, UserID: principal.Key.UserID, TeamID: principal.Key.TeamID,
		Model: model, StatusCode: status, PromptTokens: prompt, CompletionTokens: completion,
		TotalTokens: total, LatencyMS: time.Since(started).Milliseconds(), TTFTMS: ttft, ErrorCode: errorCode, Metadata: metadata,
	}); err != nil {
		log.Printf("access gateway usage persistence failed for request %s: %v", requestID, err)
	}
}
