/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// req_filter_skip_processing.go owns the request-time opt-out signal carried
// by the x-vsr-skip-processing header. The actual header capture lives in
// processor_req_header.go (the request-extraction seam); the helpers below
// build the plain CONTINUE responses that every phase must return when
// SkipProcessing is set so the request and the upstream response flow through
// untouched.
//
// Background and contract: see the VSRSkipProcessing constant in pkg/headers
// and https://github.com/vllm-project/semantic-router/issues/1808.

// newContinueRequestBodyResponse returns a plain CONTINUE response for the
// request-body phase with no body or header mutation. Used by the
// SkipProcessing passthrough so classification, routing, plugins, cache, and
// memory paths are completely bypassed.
func newContinueRequestBodyResponse() *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}
}

// handleSkipProcessingResponseHeaders short-circuits the response-header phase
// when the request was opted out via x-vsr-skip-processing. It returns a plain
// CONTINUE response (with the streaming mode override preserved when the
// upstream is SSE) so no VSR headers are added and no response inspection
// runs. Returns nil when the request did not opt out.
func (r *OpenAIRouter) handleSkipProcessingResponseHeaders(
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if ctx == nil || !ctx.SkipProcessing {
		return nil
	}
	if v != nil && v.ResponseHeaders != nil && v.ResponseHeaders.Headers != nil {
		ctx.IsStreamingResponse = isStreamingContentType(v.ResponseHeaders.Headers)
	}
	return buildResponseHeadersContinueResponse(nil, ctx.IsStreamingResponse)
}

// handleSkipProcessingResponseBody short-circuits the response-body phase for
// opted-out requests. It returns a plain CONTINUE body response so streaming
// accumulation, hallucination/jailbreak detection, cache writes, and memory
// extraction never run. Returns nil when the request did not opt out.
//
// The unused parameter mirrors handleLooperResponseBody's signature so future
// helpers can keep the same shape if they need to inspect the body.
func (r *OpenAIRouter) handleSkipProcessingResponseBody(
	_ []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if ctx == nil || !ctx.SkipProcessing {
		return nil
	}
	return buildResponseBodyContinueResponse(nil, nil)
}
