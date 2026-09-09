package extproc

import (
	"context"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// handleRequestHeaders processes the request headers.
func (r *OpenAIRouter) handleRequestHeaders(v *ext_proc.ProcessingRequest_RequestHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	ctx.StartTime = time.Now()

	span := startRequestHeaderSpan(v, ctx)
	defer span.End()

	method, path := captureRequestHeaders(v, ctx, r.skipProcessingEnabled())

	setRequestHeaderSpanAttributes(span, ctx, method, path)
	detectSourceFormat(path, ctx)
	r.deriveTrustedIdentity(ctx)
	r.applyIdentityHeaderPolicy(ctx)
	applyHeaderPassThroughPolicy(ctx)

	// Router Replay contains captured request, response, and tool data. It is a
	// management API only: the public inference listener must fail closed even
	// when a caller supplies the otherwise valid skip-processing opt-out.
	if isRouterReplayRequestTarget(path) {
		return r.createErrorResponse(404, "endpoint not found"), nil
	}

	// Honor x-vsr-skip-processing as early as possible: once captured we bypass
	// every router-side header check (replay API, validation, response-API
	// translation) and emit a plain CONTINUE so the request flows through.
	// Streaming detection still runs because the same flag drives mode selection
	// for downstream filters and is cheap; the body and response handlers will
	// also short-circuit in the no-op path.
	if ctx.SkipProcessing {
		detectStreamingExpectation(ctx)
		return newContinueRequestHeadersResponse(&ext_proc.HeaderMutation{
			RemoveHeaders: r.requestHeadersToRemove(),
		}), nil
	}

	detectStreamingExpectation(ctx)
	if modelsResp, err := r.handleModelsRequestHeaders(method, path); err != nil || modelsResp != nil {
		return modelsResp, err
	}
	if responseAPIResp, err := r.handleResponseAPIRequestHeaders(method, path, ctx); err != nil || responseAPIResp != nil {
		return responseAPIResp, err
	}
	if validationResp := r.validateRequestHeaders(method, path); validationResp != nil {
		return validationResp, nil
	}
	return newContinueRequestHeadersResponse(r.buildIdentityEncodingRequestMutation()), nil
}

func startRequestHeaderSpan(
	v *ext_proc.ProcessingRequest_RequestHeaders,
	ctx *RequestContext,
) trace.Span {
	baseCtx := ctx.TraceContext
	if baseCtx == nil {
		baseCtx = context.Background()
	}
	headerMap := make(map[string]string, len(v.RequestHeaders.Headers.Headers))
	for _, header := range v.RequestHeaders.Headers.Headers {
		headerMap[header.Key] = extractHeaderValue(header)
	}

	ctx.TraceContext = tracing.ExtractTraceContext(baseCtx, headerMap)
	spanCtx, span := tracing.StartSpan(
		ctx.TraceContext,
		tracing.SpanRequestReceived,
		trace.WithSpanKind(trace.SpanKindServer),
	)
	ctx.TraceContext = spanCtx
	return span
}

func captureRequestHeaders(
	v *ext_proc.ProcessingRequest_RequestHeaders,
	ctx *RequestContext,
	skipProcessingGateEnabled bool,
) (string, string) {
	requestHeaders := v.RequestHeaders.Headers
	for _, header := range requestHeaders.Headers {
		headerValue := extractHeaderValue(header)
		ctx.Headers[header.Key] = headerValue

		// HTTP/2 lowercases header names, but we accept either case for the
		// external skip-processing opt-out so upstream filters do not have to
		// worry about casing. Internal request authentication runs after all
		// headers have been captured so header order cannot affect validation.
		lowerKey := strings.ToLower(header.Key)
		if lowerKey == headers.RequestID {
			ctx.RequestID = headerValue
		}
		// The x-vsr-skip-processing opt-out is gated by the deployment-level
		// global.router.skip_processing.enabled flag. When disabled (the
		// default), the header is ignored entirely so an unauthenticated
		// upstream caller cannot bypass router policy by injecting it.
		if skipProcessingGateEnabled &&
			lowerKey == headers.VSRSkipProcessing &&
			strings.EqualFold(strings.TrimSpace(headerValue), "true") {
			ctx.SkipProcessing = true
		}
	}
	authenticateLooperRequestContext(ctx)

	method := ctx.Headers[":method"]
	path := ctx.Headers[":path"]
	logging.ComponentDebugEvent("extproc", "request_headers_captured", map[string]interface{}{
		"request_id":      ctx.RequestID,
		"method":          method,
		"path":            path,
		"header_count":    len(requestHeaders.Headers),
		"looper_request":  ctx.LooperRequest,
		"skip_processing": ctx.SkipProcessing,
	})

	return method, path
}

func setRequestHeaderSpanAttributes(
	span trace.Span,
	ctx *RequestContext,
	method string,
	path string,
) {
	if ctx.RequestID != "" {
		tracing.SetSpanAttributes(
			span,
			attribute.String(tracing.AttrRequestID, ctx.RequestID),
		)
	}

	tracing.SetSpanAttributes(
		span,
		attribute.String(tracing.AttrHTTPMethod, method),
		attribute.String(tracing.AttrHTTPPath, path),
	)
}

func detectStreamingExpectation(ctx *RequestContext) {
	accept, ok := ctx.Headers["accept"]
	if !ok {
		return
	}

	if strings.Contains(strings.ToLower(accept), "text/event-stream") {
		ctx.ExpectStreamingResponse = true
		logging.ComponentDebugEvent("extproc", "streaming_expectation_detected", map[string]interface{}{
			"request_id": ctx.RequestID,
			"source":     "accept_header",
		})
	}
}

func extractHeaderValue(header interface {
	GetValue() string
	GetRawValue() []byte
},
) string {
	headerValue := header.GetValue()
	if headerValue == "" && len(header.GetRawValue()) > 0 {
		return string(header.GetRawValue())
	}
	return headerValue
}

// headerValueCI is intentionally kept in the ingress header phase. It is the
// only helper allowed to translate captured request headers into typed request
// identity; downstream consumers use RequestContext.TrustedIdentity instead.
func headerValueCI(ctx *RequestContext, canonical string) string {
	if ctx == nil || len(ctx.Headers) == 0 || canonical == "" {
		return ""
	}
	if v, ok := ctx.Headers[canonical]; ok && v != "" {
		return v
	}
	for k, v := range ctx.Headers {
		if strings.EqualFold(k, canonical) && v != "" {
			return v
		}
	}
	return ""
}

func (r *OpenAIRouter) buildIdentityEncodingRequestMutation() *ext_proc.HeaderMutation {
	return &ext_proc.HeaderMutation{
		SetHeaders: []*core.HeaderValueOption{{
			Header: &core.HeaderValue{
				Key:   "accept-encoding",
				Value: "identity",
			},
		}},
		RemoveHeaders: r.requestHeadersToRemove(),
	}
}

func (r *OpenAIRouter) requestHeadersToRemove() []string {
	return appendUniqueHeaderNames(
		looperInternalHeadersForRemoval(),
		r.identityHeaderNames()...,
	)
}

// identityHeaderNames returns every authenticated identity header name
// understood by the router. The built-in names remain included so a client
// cannot bypass a configured custom name by supplying the default header.
func (r *OpenAIRouter) identityHeaderNames() []string {
	names := []string{
		headers.AuthzUserID,
		headers.AuthzUserGroups,
		headers.AuthzTenantID,
		headers.AuthzTeamID,
	}
	if r != nil && r.Config != nil {
		names = append(names,
			r.Config.Authz.Identity.GetUserIDHeader(),
			r.Config.Authz.Identity.GetUserGroupsHeader(),
		)
	}
	return appendUniqueHeaderNames(nil, names...)
}

// deriveTrustedIdentity is the single request identity extraction point. It
// runs immediately after header capture, before routing or plugin code can
// observe the request. Authenticated claims are accepted only when the
// deployment explicitly declares an external header-injection boundary;
// Router Learning continuity IDs are typed ingress values but are not auth
// claims.
func (r *OpenAIRouter) deriveTrustedIdentity(ctx *RequestContext) {
	if ctx == nil {
		return
	}

	identity := authz.TrustedIdentity{}
	learningCfg := config.RouterLearningProtectionConfig{}
	if r != nil && r.Config != nil {
		learningCfg = r.Config.RouterLearning.Protection
	}
	// x-session-id remains the explicit application/gateway override. A
	// configured learning header supplies the alternate ingress identity when
	// that override is absent; both are captured once into the typed snapshot.
	identity.SessionID = strings.TrimSpace(headerValueCI(ctx, headers.XSessionID))
	if identity.SessionID == "" {
		identity.SessionID = strings.TrimSpace(headerValueCI(ctx, learningCfg.HeaderName("session")))
	}
	identity.ConversationID = strings.TrimSpace(headerValueCI(ctx, learningCfg.HeaderName("conversation")))
	if identity.SessionID == "" {
		identity.SessionID = strings.TrimSpace(headerValueCI(ctx, headers.XClaudeCodeSessionID))
	}

	if r == nil || r.Config == nil || !r.Config.Authz.HasExternalAuthProvider() {
		ctx.TrustedIdentity = identity
		return
	}

	identity.UserID = strings.TrimSpace(headerValueCI(ctx, r.Config.Authz.Identity.GetUserIDHeader()))
	identity.Groups = parseTrustedIdentityGroups(
		headerValueCI(ctx, r.Config.Authz.Identity.GetUserGroupsHeader()),
	)
	identity.TenantID = strings.TrimSpace(headerValueCI(ctx, headers.AuthzTenantID))
	identity.TeamID = strings.TrimSpace(headerValueCI(ctx, headers.AuthzTeamID))
	ctx.TrustedIdentity = identity
}

func parseTrustedIdentityGroups(value string) []string {
	var groups []string
	for _, group := range strings.Split(value, ",") {
		if group = strings.TrimSpace(group); group != "" {
			groups = append(groups, group)
		}
	}
	return groups
}

// applyIdentityHeaderPolicy removes identity headers from the Router's
// semantic request view unless an explicit header-injection provider is
// configured. The provider declaration is the trust boundary: fail-open
// behavior or a static-config provider must not make client headers trusted.
func (r *OpenAIRouter) applyIdentityHeaderPolicy(ctx *RequestContext) {
	if ctx == nil || ctx.Headers == nil {
		return
	}
	if r != nil && r.Config != nil && r.Config.Authz.HasExternalAuthProvider() {
		return
	}

	stripped := make([]string, 0)
	for _, name := range r.identityHeaderNames() {
		for key := range ctx.Headers {
			if !strings.EqualFold(key, name) {
				continue
			}
			delete(ctx.Headers, key)
			stripped = appendUniqueHeaderNames(stripped, key)
		}
	}
	if len(stripped) == 0 {
		return
	}

	logging.ComponentWarnEvent("extproc", "untrusted_identity_headers_removed", map[string]interface{}{
		"request_id": ctx.RequestID,
		"headers":    stripped,
		"reason":     "no_explicit_external_auth_provider",
	})
}

func appendUniqueHeaderNames(names []string, additions ...string) []string {
	seen := make(map[string]struct{}, len(names)+len(additions))
	for _, name := range names {
		if name != "" {
			seen[strings.ToLower(name)] = struct{}{}
		}
	}
	for _, name := range additions {
		if name == "" {
			continue
		}
		key := strings.ToLower(name)
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		names = append(names, name)
	}
	return names
}

// hopByHopDropList is the set of HTTP framing headers we strip from
// ctx.Headers before any downstream filter or body-phase routing sees
// them. Envoy already strips most of these from the request before
// extproc receives it; we re-apply the policy as defense-in-depth and
// to make the contract explicit in code.
var hopByHopDropList = []string{
	"host",
	"content-length",
	"connection",
	"keep-alive",
	"proxy-connection",
	"transfer-encoding",
	"upgrade",
	"te",
	"trailer",
	"expect",
}

// applyHeaderPassThroughPolicy enforces the request-header pass-through
// contract by stripping transport framing from the semantic request view.
// Provider headers are supplied by the selected provider profile rather than
// copied from an untrusted client request. Identity headers are handled by
// applyIdentityHeaderPolicy because their treatment depends on authz config.
func applyHeaderPassThroughPolicy(ctx *RequestContext) {
	if ctx == nil || ctx.Headers == nil {
		return
	}

	for _, name := range hopByHopDropList {
		delete(ctx.Headers, name)
	}
}

func newContinueRequestHeadersResponse(headerMutation ...*ext_proc.HeaderMutation) *ext_proc.ProcessingResponse {
	var mutation *ext_proc.HeaderMutation
	if len(headerMutation) > 0 {
		mutation = headerMutation[0]
	}
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestHeaders{
			RequestHeaders: &ext_proc.HeadersResponse{
				Response: &ext_proc.CommonResponse{
					Status:         ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: mutation,
				},
			},
		},
	}
}
