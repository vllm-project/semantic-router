package extproc

import (
	"net/http"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// The router instance is leased for the whole extproc stream. Reading its
// config here binds a request to that generation, never a mutable API
// snapshot. The hash it was bound to rides in x-vsr-trace, its receipt.
func (r *OpenAIRouter) benchmarkConfigPrecondition(ctx *RequestContext) *ext_proc.ProcessingResponse {
	expected := headerValueCI(ctx, headers.VSRBenchExpectedConfigHash)
	cap := headerValueCI(ctx, headers.VSRBenchMaxInferenceCalls)
	if cap != "" && (cap != "1" || expected == "" || ctx.SkipProcessing) {
		return r.createErrorResponse(http.StatusBadRequest, "inference call limit requires config binding and the supported direct-call limit of 1")
	}
	if expected == "" {
		return nil
	}
	if !headers.ValidConfigHash(expected) {
		return r.createErrorResponse(http.StatusBadRequest, "expected config hash must be a lowercase SHA-256 digest")
	}
	if r.Config == nil || !headers.ValidConfigHash(r.Config.DocumentHash) {
		return r.createErrorResponse(http.StatusServiceUnavailable, "active runtime config hash is unavailable")
	}
	ctx.BenchmarkConfigHash = r.Config.DocumentHash
	if r.Config.DocumentHash != expected {
		return r.createErrorResponse(http.StatusPreconditionFailed, "active runtime config does not match the evaluation manifest")
	}
	return nil
}

// recordBenchmarkInferenceCall counts the one provider call a config-bound
// evaluation request made, when the selected model answered it and the
// route is known to make no other call.
func (r *OpenAIRouter) recordBenchmarkInferenceCall(ctx *RequestContext, successful bool) {
	if successful && ctx.BenchmarkConfigHash != "" && ctx.BenchmarkModelUsage == "" && !ctx.VSRCacheHit &&
		ctx.VSRSelectedModel != "" && r.benchmarkUsageScopeKnown(ctx, false) {
		ctx.BenchmarkInferenceCalls = 1
	}
}

// bindBenchmarkConfigResponse keeps Envoy from retrying or hedging the
// provider call of a config-bound evaluation request, so its receipt counts
// every call the request made.
func (r *OpenAIRouter) bindBenchmarkConfigResponse(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	if response == nil || headerValueCI(ctx, headers.VSRBenchExpectedConfigHash) == "" {
		return
	}
	var common *ext_proc.CommonResponse
	switch output := response.Response.(type) {
	case *ext_proc.ProcessingResponse_RequestHeaders:
		common = output.RequestHeaders.GetResponse()
	case *ext_proc.ProcessingResponse_RequestBody:
		common = output.RequestBody.GetResponse()
	}
	if common == nil {
		return
	}
	if common.HeaderMutation == nil {
		common.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	setHeaderValue(common.HeaderMutation, headers.EnvoyMaxRetries, "0")
	setHeaderValue(common.HeaderMutation, headers.EnvoyHedgeOnPerTryTimeout, "false")
}
