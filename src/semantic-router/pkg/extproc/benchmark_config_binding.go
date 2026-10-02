package extproc

import (
	"net/http"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// The router instance is leased for the whole extproc stream. Reading its
// config here binds a request to that generation, never a mutable API snapshot.
func (r *OpenAIRouter) benchmarkConfigPrecondition(ctx *RequestContext) *ext_proc.ProcessingResponse {
	expected := headerValueCI(ctx, headers.SRBenchExpectedConfigHash)
	cap := headerValueCI(ctx, headers.SRBenchMaxInferenceCalls)
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
	if r.Config.DocumentHash != expected {
		return r.createErrorResponse(http.StatusPreconditionFailed, "active runtime config does not match the evaluation manifest")
	}
	return nil
}

// bindBenchmarkConfigResponse applies the generation receipt at the transport
// boundary, including cache, looper, streaming, and immediate error responses.
func (r *OpenAIRouter) bindBenchmarkConfigResponse(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	if response == nil || headerValueCI(ctx, headers.SRBenchExpectedConfigHash) == "" {
		return
	}
	var mutation **ext_proc.HeaderMutation
	switch output := response.Response.(type) {
	case *ext_proc.ProcessingResponse_RequestHeaders:
		if output.RequestHeaders != nil && output.RequestHeaders.Response != nil {
			if output.RequestHeaders.Response.HeaderMutation == nil {
				output.RequestHeaders.Response.HeaderMutation = &ext_proc.HeaderMutation{}
			}
			output.RequestHeaders.Response.HeaderMutation.RemoveHeaders = append(
				output.RequestHeaders.Response.HeaderMutation.RemoveHeaders,
				headers.SRBenchExpectedConfigHash, headers.SRBenchMaxInferenceCalls, headers.VSRConfigHash, headers.VSRModelUsage, headers.VSRInferenceCallCount,
			)
			setHeaderValue(output.RequestHeaders.Response.HeaderMutation, "x-envoy-max-retries", "0")
			setHeaderValue(output.RequestHeaders.Response.HeaderMutation, "x-envoy-hedge-on-per-try-timeout", "false")
		}
		return
	case *ext_proc.ProcessingResponse_RequestBody:
		if output.RequestBody != nil && output.RequestBody.Response != nil {
			if output.RequestBody.Response.HeaderMutation == nil {
				output.RequestBody.Response.HeaderMutation = &ext_proc.HeaderMutation{}
			}
			output.RequestBody.Response.HeaderMutation.RemoveHeaders = append(output.RequestBody.Response.HeaderMutation.RemoveHeaders, headers.SRBenchExpectedConfigHash, headers.SRBenchMaxInferenceCalls, headers.VSRConfigHash, headers.VSRModelUsage, headers.VSRInferenceCallCount)
			setHeaderValue(output.RequestBody.Response.HeaderMutation, "x-envoy-max-retries", "0")
			setHeaderValue(output.RequestBody.Response.HeaderMutation, "x-envoy-hedge-on-per-try-timeout", "false")
		}
		return
	case *ext_proc.ProcessingResponse_ResponseHeaders:
		if output.ResponseHeaders != nil && output.ResponseHeaders.Response != nil {
			mutation = &output.ResponseHeaders.Response.HeaderMutation
		}
	case *ext_proc.ProcessingResponse_ImmediateResponse:
		if output.ImmediateResponse != nil {
			mutation = &output.ImmediateResponse.Headers
		}
	}
	if mutation == nil || r.Config == nil || !headers.ValidConfigHash(r.Config.DocumentHash) {
		return
	}
	if *mutation == nil {
		*mutation = &ext_proc.HeaderMutation{}
	}
	// Replace any upstream or cached value; append would permit a forged receipt.
	kept := (*mutation).SetHeaders[:0]
	for _, option := range (*mutation).SetHeaders {
		if option.GetHeader() == nil || !benchmarkReceiptHeader(option.GetHeader().GetKey()) {
			kept = append(kept, option)
		}
	}
	(*mutation).RemoveHeaders = append((*mutation).RemoveHeaders, headers.VSRModelUsage, headers.VSRInferenceCallCount)
	(*mutation).SetHeaders = kept
	(*mutation).SetHeaders = append((*mutation).SetHeaders, &core.HeaderValueOption{
		Header:       &core.HeaderValue{Key: headers.VSRConfigHash, RawValue: []byte(r.Config.DocumentHash)},
		AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
	})
	if ctx.BenchmarkModelUsage != "" {
		setHeaderValue(*mutation, headers.VSRModelUsage, ctx.BenchmarkModelUsage)
	} else if response.GetResponseHeaders() != nil && ctx.UpstreamStatusCode >= 200 && ctx.UpstreamStatusCode < 300 && ctx.VSRSelectedModel != "" && !ctx.VSRCacheHit && r.benchmarkUsageScopeKnown(ctx, false) {
		setHeaderValue(*mutation, headers.VSRInferenceCallCount, "1")
	}
}

func benchmarkReceiptHeader(name string) bool {
	return strings.EqualFold(name, headers.VSRConfigHash) || strings.EqualFold(name, headers.VSRModelUsage) || strings.EqualFold(name, headers.VSRInferenceCallCount)
}
