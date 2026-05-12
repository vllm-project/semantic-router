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

func (r *OpenAIRouter) skipProcessingEnabled() bool {
	if r == nil || r.Config == nil {
		return false
	}
	return r.Config.SkipProcessing.IsEnabled()
}

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

func (r *OpenAIRouter) handleSkipProcessingResponseBody(
	_ []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if ctx == nil || !ctx.SkipProcessing {
		return nil
	}
	return buildResponseBodyContinueResponse(nil, nil)
}
