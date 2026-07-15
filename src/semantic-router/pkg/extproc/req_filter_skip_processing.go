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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func (r *OpenAIRouter) skipProcessingEnabled() bool {
	if r == nil || r.Config == nil {
		return false
	}
	return r.Config.SkipProcessing.IsEnabled()
}

func (r *OpenAIRouter) newContinueRequestBodyResponse() *ext_proc.ProcessingResponse {
	// Skip-processing bypasses routing entirely, so there is no backend profile
	// to opt into forward_authorization_header. Strip the caller's Authorization,
	// the internal looper carrier, and every ext_authz-injected per-user key (the
	// same set the routing path removes via CredentialResolver.HeadersToStrip) so
	// an opted-out request can never leak a caller credential to whatever backend
	// Envoy routes it to. See issue #2375.
	removeHeaders := []string{
		forwardedAuthorizationHeaderName,
		headers.VSRInboundAuthorization,
	}
	if r != nil && r.CredentialResolver != nil {
		removeHeaders = append(removeHeaders, r.CredentialResolver.HeadersToStrip()...)
	}
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: &ext_proc.HeaderMutation{
						RemoveHeaders: removeHeaders,
					},
				},
			},
		},
	}
}

func newFullDuplexRequestBodyResponse(body []byte, endOfStream bool) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_StreamedResponse{
							StreamedResponse: &ext_proc.StreamedBodyResponse{
								Body:        body,
								EndOfStream: endOfStream,
							},
						},
					},
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
