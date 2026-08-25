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

// stripInboundAuthorizationEnabled reports whether the opt-in hardening that
// strips the caller's inbound Authorization (and ext_authz-injected per-user
// keys) on the default/passthrough path is enabled. Off by default so the
// caller's Authorization is preserved unless an operator opts in via
// global.router.strip_inbound_authorization. See issue #2375.
func (r *OpenAIRouter) stripInboundAuthorizationEnabled() bool {
	if r == nil || r.Config == nil {
		return false
	}
	return r.Config.StripInboundAuthorization
}

func (r *OpenAIRouter) newContinueRequestBodyResponse() *ext_proc.ProcessingResponse {
	// The internal looper carrier for caller identity must never reach an upstream
	// (unconditional — it is a router-internal header, not a caller credential).
	removeHeaders := []string{
		headers.VSRInboundAuthorization,
	}
	// Opt-in hardening (global.router.strip_inbound_authorization): skip-processing
	// bypasses routing entirely, so there is no backend profile to inject a static
	// key or opt into forward_authorization_header. When enabled, strip the
	// caller's Authorization and every ext_authz-injected per-user key (the same
	// set the routing path removes via CredentialResolver.HeadersToStrip) so an
	// opted-out request cannot leak a caller credential to whatever backend Envoy
	// routes it to. Off by default, preserving the caller's Authorization on the
	// passthrough. See issue #2375.
	if r.stripInboundAuthorizationEnabled() {
		// stripInboundAuthorizationEnabled already guaranteed r is non-nil.
		removeHeaders = append(removeHeaders, forwardedAuthorizationHeaderName)
		if r.CredentialResolver != nil {
			removeHeaders = append(removeHeaders, r.CredentialResolver.HeadersToStrip()...)
		}
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
