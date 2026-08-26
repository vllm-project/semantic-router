package backendinvoker

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const DispatchCapabilityHeader = "X-VSR-Dispatch-Capability"

type PlanResolver interface {
	ResolvePlans(context.Context, DispatchCapability) (PlanChain, error)
}

type ResponseObserver interface {
	Observe(context.Context, Plan, AttemptResult, *http.Response) (io.ReadCloser, error)
}

type Handler struct {
	Audience       string
	Keyring        SigningKeyring
	Plans          PlanResolver
	Invoker        *Invoker
	Observer       ResponseObserver
	MaxRequestBody int64
	Now            func() time.Time
}

type preparedDispatchRequest struct {
	capability DispatchCapability
	plans      PlanChain
	now        time.Time
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, request *http.Request) {
	if h == nil || h.Plans == nil || h.Invoker == nil || h.Observer == nil {
		http.Error(w, "backend invoker is unavailable", http.StatusServiceUnavailable)
		return
	}
	prepared, ok := h.prepareDispatchRequest(w, request)
	if !ok {
		return
	}
	h.invokePreparedRequest(w, request, prepared)
}

func (h *Handler) prepareDispatchRequest(
	w http.ResponseWriter, request *http.Request,
) (preparedDispatchRequest, bool) {
	now := time.Now
	if h.Now != nil {
		now = h.Now
	}
	verifiedAt := now().UTC()
	capability, serveHTTPErr := h.Keyring.Verify(strings.TrimSpace(request.Header.Get(DispatchCapabilityHeader)), h.Audience, verifiedAt)
	if serveHTTPErr != nil {
		http.Error(w, "invalid dispatch capability", http.StatusUnauthorized)
		return preparedDispatchRequest{}, false
	}
	request.Header.Del(DispatchCapabilityHeader)
	limit := h.MaxRequestBody
	if limit <= 0 {
		limit = 64 << 20
	}
	body, serveHTTPErr := io.ReadAll(io.LimitReader(request.Body, limit+1))
	if serveHTTPErr != nil || int64(len(body)) > limit {
		h.writePreDispatchError(w, capability, verifiedAt, http.StatusRequestEntityTooLarge,
			llmprotocol.ErrorInvalidRequest, "request_too_large", "request body is too large", serveHTTPErr)
		return preparedDispatchRequest{}, false
	}
	if request.Method != capability.Method || request.URL.Path != capability.Path ||
		request.URL.RawQuery != capability.Query {
		h.writePreDispatchError(w, capability, verifiedAt, http.StatusUnauthorized,
			llmprotocol.ErrorAuthentication, "dispatch_mismatch", "dispatch request could not be authenticated", nil)
		return preparedDispatchRequest{}, false
	}
	digest := RequestDigest(request.Method, request.URL.Path, request.URL.RawQuery, body)
	if len(digest) != len(capability.RequestDigest) || subtle.ConstantTimeCompare([]byte(digest), []byte(capability.RequestDigest)) != 1 {
		h.writePreDispatchError(w, capability, verifiedAt, http.StatusUnauthorized,
			llmprotocol.ErrorAuthentication, "dispatch_mismatch", "dispatch request could not be authenticated", nil)
		return preparedDispatchRequest{}, false
	}
	plans, serveHTTPErr := h.Plans.ResolvePlans(request.Context(), capability)
	if serveHTTPErr != nil {
		h.writePreDispatchError(w, capability, verifiedAt, http.StatusNotFound,
			llmprotocol.ErrorNotFound, "dispatch_target_unavailable", "dispatch target is unavailable", serveHTTPErr)
		return preparedDispatchRequest{}, false
	}
	if err := bindCapabilityToPlans(&plans, capability); err != nil {
		h.writePreDispatchError(w, capability, verifiedAt, http.StatusUnauthorized,
			llmprotocol.ErrorAuthentication, "dispatch_target_mismatch", "dispatch target could not be authenticated", err)
		return preparedDispatchRequest{}, false
	}
	streaming := requestUsesStreaming(body)
	for index := range plans.Candidates {
		plan := &plans.Candidates[index]
		plan.Method = request.Method
		plan.Path = request.URL.Path
		plan.Query = request.URL.RawQuery
		plan.Headers = request.Header.Clone()
		plan.Body = append([]byte(nil), body...)
		plan.RequestDigest = capability.RequestDigest
		plan.Streaming = streaming
		plan.SourceFormat = capability.WireFormat
	}
	return preparedDispatchRequest{capability: capability, plans: plans, now: now().UTC()}, true
}

// writePreDispatchError proves that a verified capability reached a terminal
// private-handler failure before any physical backend attempt began. The
// signed empty outcome is request-bound evidence; callers must not infer
// known-zero usage from an unsigned error response.
func (h *Handler) writePreDispatchError(
	w http.ResponseWriter,
	capability DispatchCapability,
	now time.Time,
	status int,
	category llmprotocol.ErrorCategory,
	code, message string,
	cause error,
) {
	if err := setDispatchOutcome(w.Header(), h.Keyring, capability, Result{}, now); err != nil {
		h.writeWireError(w, capability.WireFormat, http.StatusServiceUnavailable,
			llmprotocol.ErrorInternal, "dispatch_outcome_unavailable", "dispatch outcome is unavailable", err)
		return
	}
	h.writeWireError(w, capability.WireFormat, status, category, code, message, cause)
}

func (h *Handler) invokePreparedRequest(
	w http.ResponseWriter, request *http.Request, prepared preparedDispatchRequest,
) {
	result, serveHTTPErr := h.Invoker.InvokeChain(request.Context(), prepared.plans)
	if serveHTTPErr != nil {
		if outcomeErr := setDispatchOutcome(w.Header(), h.Keyring, prepared.capability, result, prepared.now); outcomeErr != nil {
			h.writeWireError(w, prepared.capability.WireFormat, http.StatusServiceUnavailable,
				llmprotocol.ErrorInternal, "dispatch_outcome_unavailable", "dispatch outcome is unavailable", outcomeErr)
			return
		}
		status := http.StatusBadGateway
		if result.Attempt.State == AttemptUnknown {
			status = http.StatusServiceUnavailable
		}
		protocolError := failedResponseTerminal(serveHTTPErr).Error
		h.writeProtocolError(w, prepared.capability.WireFormat, status, protocolError)
		return
	}
	if result.Response == nil || result.Response.Body == nil || result.Selected == nil {
		h.writeWireError(w, prepared.capability.WireFormat, http.StatusBadGateway,
			llmprotocol.ErrorUpstreamUnavailable, "empty_upstream_response", "the selected model returned no response", nil)
		return
	}
	observed, serveHTTPErr := h.Observer.Observe(request.Context(), *result.Selected, result.Attempt, result.Response)
	if serveHTTPErr != nil {
		_ = result.Response.Body.Close()
		if outcomeErr := setDispatchOutcome(w.Header(), h.Keyring, prepared.capability, result, prepared.now); outcomeErr != nil {
			h.writeWireError(w, prepared.capability.WireFormat, http.StatusServiceUnavailable,
				llmprotocol.ErrorInternal, "dispatch_outcome_unavailable", "dispatch outcome is unavailable", outcomeErr)
			return
		}
		h.writeWireError(w, prepared.capability.WireFormat, http.StatusServiceUnavailable,
			llmprotocol.ErrorInternal, "response_accounting_unavailable", "response accounting is unavailable", serveHTTPErr)
		return
	}
	defer observed.Close()
	copyResponseHeaders(w.Header(), result.Response.Header)
	if err := setDispatchOutcome(w.Header(), h.Keyring, prepared.capability, result, prepared.now); err != nil {
		h.writeWireError(w, prepared.capability.WireFormat, http.StatusServiceUnavailable,
			llmprotocol.ErrorInternal, "dispatch_outcome_unavailable", "dispatch outcome is unavailable", err)
		return
	}
	w.WriteHeader(result.Response.StatusCode)
	if err := copyStreaming(w, observed); err != nil {
		return
	}
}

func (h *Handler) writeWireError(
	w http.ResponseWriter,
	format llmprotocol.WireFormat,
	status int,
	category llmprotocol.ErrorCategory,
	code, message string,
	cause error,
) {
	h.writeProtocolError(w, format, status, llmprotocol.NewError(category, code, message, cause))
}

func (h *Handler) writeProtocolError(w http.ResponseWriter, format llmprotocol.WireFormat, status int, protocolError *llmprotocol.ProtocolError) {
	registry := protocolcodec.NewBuiltinRegistry()
	if h != nil && h.Invoker != nil {
		registry = h.Invoker.codecRegistry()
	}
	engine, err := protocolcodec.NewEngine(registry, llmprotocol.DefaultPolicy())
	if err != nil {
		http.Error(w, "request failed", status)
		return
	}
	body, err := engine.EncodeError(format, protocolError)
	if err != nil {
		http.Error(w, "request failed", status)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	if protocolError != nil && protocolError.RetryAfter > 0 {
		w.Header().Set("Retry-After", fmt.Sprintf("%d", protocolError.RetryAfter))
	}
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

func requestUsesStreaming(body []byte) bool {
	var envelope struct {
		Stream bool `json:"stream"`
	}
	return json.Unmarshal(body, &envelope) == nil && envelope.Stream
}

func bindCapabilityToPlans(plans *PlanChain, capability DispatchCapability) error {
	if plans == nil || len(plans.Candidates) != len(capability.Candidates) ||
		!sameFallbackPolicy(plans.Fallback, capability.Fallback) {
		return fmt.Errorf("resolved plan chain does not match dispatch capability")
	}
	for index := range plans.Candidates {
		plan := &plans.Candidates[index]
		candidate := capability.Candidates[index]
		if plan.NamespaceID != capability.NamespaceID || plan.QuotaPartition != capability.QuotaPartition ||
			plan.PublicationID != capability.PublicationID || plan.RuntimeEpoch != capability.RuntimeEpoch ||
			plan.RoutingRevision != capability.RoutingRevision || plan.RoutingDigest != capability.RoutingDigest ||
			plan.AdmissionID != capability.AdmissionID || plan.AdmissionDigest != capability.AdmissionDigest ||
			plan.RequestID != capability.RequestID || !sameCandidate(candidate, candidateFromPlan(*plan)) {
			return fmt.Errorf("resolved candidate plan %d does not match dispatch capability", index)
		}
	}
	return nil
}

func sameFallbackPolicy(left, right FallbackPolicy) bool {
	if len(left.On) != len(right.On) {
		return false
	}
	for index := range left.On {
		if left.On[index] != right.On[index] {
			return false
		}
	}
	return true
}

func setDispatchOutcome(
	headers http.Header,
	keyring SigningKeyring,
	capability DispatchCapability,
	result Result,
	now time.Time,
) error {
	outcome, err := outcomeForResult(capability, result, now, keyring.MaxLifetime)
	if err != nil {
		return err
	}
	token, err := keyring.SignOutcome(outcome, now)
	if err != nil {
		return err
	}
	headers.Set(DispatchOutcomeHeader, token)
	return nil
}

func copyResponseHeaders(target, source http.Header) {
	for key, values := range source {
		canonical := strings.ToLower(strings.TrimSpace(key))
		if strings.HasPrefix(canonical, "x-vsr-") ||
			strings.HasPrefix(canonical, "x-vllm-sr-") ||
			strings.HasPrefix(canonical, "x-authz-") {
			continue
		}
		switch canonical {
		case "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade", "set-cookie", "server", "x-api-key", "api-key":
			continue
		}
		for _, value := range values {
			target.Add(key, value)
		}
	}
	target.Set("Cache-Control", "no-store")
}

func copyStreaming(w http.ResponseWriter, source io.Reader) error {
	buffer := make([]byte, 32<<10)
	flusher, canFlush := w.(http.Flusher)
	for {
		read, readErr := source.Read(buffer)
		if read > 0 {
			if _, err := w.Write(buffer[:read]); err != nil {
				return err
			}
			if canFlush {
				flusher.Flush()
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				return nil
			}
			return readErr
		}
	}
}
