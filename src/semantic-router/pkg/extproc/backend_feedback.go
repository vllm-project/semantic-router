package extproc

import (
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) captureBackendFeedback(responseHeaders *core.HeaderMap, ctx *RequestContext) {
	if ctx == nil || responseHeaders == nil {
		return
	}
	ctx.ActualBackendID = backendFeedbackHeader(responseHeaders, headers.VSRActualBackend)
	ctx.ActualReplicaID = backendFeedbackHeader(responseHeaders, headers.VSRActualReplica)
	ctx.ActualUpstream = backendFeedbackHeader(responseHeaders, headers.VSRActualUpstream)
	if ctx.ActualBackendID == "" && ctx.ActualReplicaID == "" && ctx.ActualUpstream == "" {
		return
	}
	if ctx.UpstreamSpan != nil {
		ctx.UpstreamSpan.SetAttributes(
			attribute.String("vsr.backend.actual_backend_id", ctx.ActualBackendID),
			attribute.String("vsr.backend.actual_replica_id", ctx.ActualReplicaID),
			attribute.String("vsr.backend.actual_upstream", ctx.ActualUpstream),
		)
	}

	logging.ComponentDebugEvent("extproc", "backend_feedback_received", map[string]interface{}{
		"request_id":        ctx.RequestID,
		"requested_backend": ctx.RequestedBackendID,
		"actual_backend":    ctx.ActualBackendID,
		"actual_replica":    ctx.ActualReplicaID,
		"actual_upstream":   ctx.ActualUpstream,
	})
	r.updateRouterReplayRouteDiagnostics(ctx)
}

func backendFeedbackHeader(headerMap *core.HeaderMap, key string) string {
	if headerMap == nil {
		return ""
	}
	for _, header := range headerMap.Headers {
		if !strings.EqualFold(header.Key, key) {
			continue
		}
		return strings.TrimSpace(extractHeaderValue(header))
	}
	return ""
}

func (r *OpenAIRouter) updateRouterReplayRouteDiagnostics(ctx *RequestContext) {
	if ctx == nil || ctx.RouterReplayID == "" {
		return
	}
	recorder := ctx.RouterReplayRecorder
	if recorder == nil {
		recorder = r.ReplayRecorder
	}
	if recorder == nil {
		return
	}

	record, found := recorder.GetRecord(ctx.RouterReplayID)
	if !found || record.RouteDiagnostics == nil {
		return
	}
	diagnostics := *record.RouteDiagnostics
	applyBackendRouteDiagnostics(ctx, &diagnostics)
	if err := recorder.UpdateRouteDiagnostics(ctx.RouterReplayID, diagnostics); err != nil {
		logging.ComponentErrorEvent("extproc", "router_replay_route_diagnostics_update_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
	}
}
