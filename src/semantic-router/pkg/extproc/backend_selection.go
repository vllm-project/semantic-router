package extproc

import (
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backend"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	backendSelectionModeRequested = "requested"
	backendSelectionModeFailOpen  = "fail_open"
)

func internalBackendHeaders() []string {
	return []string{
		headers.VSRSelectedBackend,
		headers.VSRSelectedReplica,
		headers.VSRActualBackend,
		headers.VSRActualReplica,
		headers.VSRActualUpstream,
	}
}

func stripInternalBackendHeaders(ctx *RequestContext) {
	if ctx == nil || ctx.Headers == nil {
		return
	}
	for _, name := range internalBackendHeaders() {
		delete(ctx.Headers, name)
	}
}

func (r *OpenAIRouter) appendBackendSelectionHeaders(
	setHeaders *[]*core.HeaderValueOption,
	model string,
	ctx *RequestContext,
) {
	result := r.selectBackendCandidate(model, ctx)
	if result.FailOpen {
		setBackendSpanAttributes(ctx, result)
		return
	}
	appendHeaderRaw(setHeaders, headers.VSRSelectedBackend, result.SelectedBackendID)
	appendHeaderRaw(setHeaders, headers.VSRSelectedReplica, result.SelectedReplicaID)
	setBackendSpanAttributes(ctx, result)
}

func (r *OpenAIRouter) selectBackendCandidate(model string, ctx *RequestContext) backend.BackendPolicyResult {
	candidates := backendCandidatesForModel(r.Config, model)
	result := backend.SelectBackendCandidate(model, candidates, backend.DefaultStore())
	applyBackendPolicyResult(ctx, result)

	if ctx != nil {
		logging.ComponentDebugEvent("extproc", "backend_policy_evaluated", map[string]interface{}{
			"request_id":            ctx.RequestID,
			"model":                 model,
			"selected_backend_id":   result.SelectedBackendID,
			"selected_replica_id":   result.SelectedReplicaID,
			"fail_open":             result.FailOpen,
			"reason":                result.Reason,
			"candidate_count":       result.Diagnostics.CandidateCount,
			"fresh_candidate_count": result.Diagnostics.FreshCandidateCount,
		})
	}
	return result
}

func applyBackendPolicyResult(ctx *RequestContext, result backend.BackendPolicyResult) {
	if ctx == nil {
		return
	}
	ctx.BackendPolicyReason = result.Reason
	ctx.BackendFallbackReason = result.Diagnostics.FallbackReason
	ctx.RequestedBackendID = ""
	ctx.RequestedReplicaID = ""
	if result.FailOpen {
		return
	}
	ctx.RequestedBackendID = result.SelectedBackendID
	ctx.RequestedReplicaID = result.SelectedReplicaID
}

func backendCandidatesForModel(cfg *config.RouterConfig, model string) []backend.BackendCandidate {
	if cfg == nil {
		return nil
	}
	endpoints := cfg.GetEndpointsForModel(model)
	candidates := make([]backend.BackendCandidate, 0, len(endpoints))
	for _, endpoint := range endpoints {
		backendID := strings.TrimSpace(endpoint.BackendID)
		if backendID == "" {
			backendID = strings.TrimSpace(endpoint.Name)
		}
		if backendID == "" {
			continue
		}
		weight := endpoint.Weight
		if weight <= 0 {
			weight = 1
		}
		candidates = append(candidates, backend.BackendCandidate{
			BackendID:    backendID,
			ModelName:    model,
			EndpointName: endpoint.Name,
			Weight:       weight,
		})
	}
	return candidates
}

func appendHeaderRaw(setHeaders *[]*core.HeaderValueOption, key string, value string) {
	if strings.TrimSpace(value) == "" {
		return
	}
	*setHeaders = append(*setHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      key,
			RawValue: []byte(value),
		},
	})
}

func setBackendSpanAttributes(ctx *RequestContext, result backend.BackendPolicyResult) {
	if ctx == nil || ctx.UpstreamSpan == nil {
		return
	}
	ctx.UpstreamSpan.SetAttributes(
		attribute.Bool("vsr.backend.fail_open", result.FailOpen),
		attribute.String("vsr.backend.reason", result.Reason),
		attribute.String("vsr.backend.selected_backend_id", result.SelectedBackendID),
		attribute.String("vsr.backend.selected_replica_id", result.SelectedReplicaID),
	)
}
