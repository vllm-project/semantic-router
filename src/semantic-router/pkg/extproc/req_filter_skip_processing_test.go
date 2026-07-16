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
	"slices"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func newRouterWithSkipProcessingGate(enabled bool) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterOptions: config.RouterOptions{
				SkipProcessing: config.SkipProcessingConfig{Enabled: enabled},
			},
		},
	}
}

func newSkipProcessingRequestHeaders(method, path, value string) *ext_proc.ProcessingRequest_RequestHeaders {
	return &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":method", Value: method},
					{Key: ":path", Value: path},
					{Key: headers.VSRSkipProcessing, Value: value},
				},
			},
		},
	}
}

func TestHandleRequestHeadersSkipProcessingOptOut(t *testing.T) {
	tests := []struct {
		name           string
		headerValue    string
		expectSkipFlag bool
	}{
		{name: "lowercase true opts out", headerValue: "true", expectSkipFlag: true},
		{name: "uppercase TRUE opts out", headerValue: "TRUE", expectSkipFlag: true},
		{name: "mixed case True opts out", headerValue: "True", expectSkipFlag: true},
		{name: "true with surrounding whitespace opts out", headerValue: "  true  ", expectSkipFlag: true},
		{name: "false does not opt out", headerValue: "false", expectSkipFlag: false},
		{name: "empty value does not opt out", headerValue: "", expectSkipFlag: false},
		{name: "non-boolean value does not opt out", headerValue: "yes", expectSkipFlag: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := newRouterWithSkipProcessingGate(true)
			ctx := &RequestContext{Headers: make(map[string]string)}

			request := newSkipProcessingRequestHeaders("POST", "/v1/chat/completions", tt.headerValue)
			response, err := router.handleRequestHeaders(request, ctx)
			if err != nil {
				t.Fatalf("handleRequestHeaders failed: %v", err)
			}
			if ctx.SkipProcessing != tt.expectSkipFlag {
				t.Fatalf("expected SkipProcessing=%v, got %v", tt.expectSkipFlag, ctx.SkipProcessing)
			}
			if response.GetRequestHeaders() == nil {
				t.Fatalf("expected continue request headers response, got %T", response.GetResponse())
			}
			if response.GetRequestHeaders().Response.Status != ext_proc.CommonResponse_CONTINUE {
				t.Fatalf("expected CONTINUE status, got %v", response.GetRequestHeaders().Response.Status)
			}
		})
	}
}

func TestHandleRequestHeadersSkipProcessingHeaderIsCaseInsensitive(t *testing.T) {
	router := newRouterWithSkipProcessingGate(true)
	ctx := &RequestContext{Headers: make(map[string]string)}

	request := &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":method", Value: "POST"},
					{Key: ":path", Value: "/v1/chat/completions"},
					{Key: "X-VSR-Skip-Processing", Value: "true"},
				},
			},
		},
	}

	response, err := router.handleRequestHeaders(request, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders failed: %v", err)
	}
	if !ctx.SkipProcessing {
		t.Fatal("expected SkipProcessing to be set when header arrives in mixed case")
	}
	if response.GetRequestHeaders() == nil {
		t.Fatal("expected continue request headers response")
	}
}

func TestHandleRequestHeadersSkipProcessingBypassesValidation(t *testing.T) {
	router := newRouterWithSkipProcessingGate(true)
	ctx := &RequestContext{Headers: make(map[string]string)}

	request := newSkipProcessingRequestHeaders("POST", "/v1/does-not-exist", "true")
	response, err := router.handleRequestHeaders(request, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders failed: %v", err)
	}
	if !ctx.SkipProcessing {
		t.Fatal("expected SkipProcessing to be set")
	}
	if response.GetImmediateResponse() != nil {
		t.Fatalf("expected validation to be skipped, got immediate response: %v", response.GetImmediateResponse().Status)
	}
	if response.GetRequestHeaders() == nil {
		t.Fatal("expected continue request headers response")
	}
}

func TestHandleRequestHeadersSkipProcessingPreservesStreamingDetection(t *testing.T) {
	router := newRouterWithSkipProcessingGate(true)
	ctx := &RequestContext{Headers: make(map[string]string)}

	request := &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":method", Value: "POST"},
					{Key: ":path", Value: "/v1/chat/completions"},
					{Key: "accept", Value: "text/event-stream"},
					{Key: headers.VSRSkipProcessing, Value: "true"},
				},
			},
		},
	}

	if _, err := router.handleRequestHeaders(request, ctx); err != nil {
		t.Fatalf("handleRequestHeaders failed: %v", err)
	}
	if !ctx.SkipProcessing {
		t.Fatal("expected SkipProcessing to be set")
	}
	if !ctx.ExpectStreamingResponse {
		t.Fatal("expected streaming expectation to still be detected even when skipping processing")
	}
}

func TestHandleRequestHeadersSkipProcessingHeaderIgnoredWhenGateDisabled(t *testing.T) {
	tests := []struct {
		name   string
		router *OpenAIRouter
	}{
		{name: "default zero-value router has gate disabled", router: &OpenAIRouter{}},
		{name: "explicit disabled gate", router: newRouterWithSkipProcessingGate(false)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: make(map[string]string)}
			request := newSkipProcessingRequestHeaders("GET", "/v1/models", "true")

			response, err := tt.router.handleRequestHeaders(request, ctx)
			if err != nil {
				t.Fatalf("handleRequestHeaders failed: %v", err)
			}
			if ctx.SkipProcessing {
				t.Fatal("expected SkipProcessing to remain false when the deployment gate is off, even with the header set to true")
			}
			if got := ctx.Headers[headers.VSRSkipProcessing]; got != "true" {
				t.Fatalf("expected raw header to still be captured, got %q", got)
			}
			if response == nil {
				t.Fatal("expected a processing response even when the header is ignored")
			}
		})
	}
}

func TestSkipProcessingEnabledHelper(t *testing.T) {
	tests := []struct {
		name   string
		router *OpenAIRouter
		want   bool
	}{
		{name: "nil router", router: nil, want: false},
		{name: "router without config", router: &OpenAIRouter{}, want: false},
		{name: "config with zero-value SkipProcessing", router: &OpenAIRouter{Config: &config.RouterConfig{}}, want: false},
		{name: "operator enabled the gate", router: newRouterWithSkipProcessingGate(true), want: true},
		{name: "operator explicitly disabled the gate", router: newRouterWithSkipProcessingGate(false), want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.router.skipProcessingEnabled(); got != tt.want {
				t.Fatalf("skipProcessingEnabled()=%v, want %v", got, tt.want)
			}
		})
	}
}

func TestHandleRequestBodyDispatchSkipProcessingShortCircuits(t *testing.T) {
	// Enable the opt-in strip so the passthrough removes the caller credential.
	router := &OpenAIRouter{Config: &config.RouterConfig{
		RouterOptions: config.RouterOptions{StripInboundAuthorization: true},
	}}
	ctx := &RequestContext{
		Headers:        make(map[string]string),
		SkipProcessing: true,
	}

	request := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{
			Body:        []byte(""),
			EndOfStream: true,
		},
	}

	response, err := router.handleRequestBodyDispatch(request, ctx)
	if err != nil {
		t.Fatalf("handleRequestBodyDispatch failed: %v", err)
	}
	if response.GetRequestBody() == nil {
		t.Fatalf("expected request body continue response, got %T", response.GetResponse())
	}
	if response.GetRequestBody().Response.Status != ext_proc.CommonResponse_CONTINUE {
		t.Fatalf("expected CONTINUE status, got %v", response.GetRequestBody().Response.Status)
	}
	if response.GetRequestBody().Response.GetBodyMutation() != nil {
		t.Fatal("expected no body mutation when skipping processing")
	}
	// Skip processing must still strip the caller credential (issue #2375): it
	// bypasses routing, so there is no backend opt-in that could authorize
	// forwarding the caller Authorization to the downstream.
	mutation := response.GetRequestBody().Response.GetHeaderMutation()
	if mutation == nil {
		t.Fatal("expected a header mutation stripping the caller credential when skipping processing")
	}
	removeHeaders := mutation.GetRemoveHeaders()
	if !slices.Contains(removeHeaders, forwardedAuthorizationHeaderName) {
		t.Fatalf("expected skip processing to strip %q, got remove_headers=%v", forwardedAuthorizationHeaderName, removeHeaders)
	}
	if !slices.Contains(removeHeaders, headers.VSRInboundAuthorization) {
		t.Fatalf("expected skip processing to strip %q, got remove_headers=%v", headers.VSRInboundAuthorization, removeHeaders)
	}
	if ctx.StreamedBody != nil {
		t.Fatal("expected no streamed body handler to be allocated when skipping processing")
	}
}

// TestSkipProcessingResponseStripsCallerAuthorization proves the skip-processing
// passthrough removes the caller credential, the internal carrier, AND the
// ext_authz-injected per-user keys (the same set the routing path strips), since
// skip bypasses routing and has no backend opt-in (issue #2375).
func TestSkipProcessingResponseStripsCallerAuthorization(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{StripInboundAuthorization: true},
	}
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: newTestCredentialResolver(cfg),
	}

	resp := router.newContinueRequestBodyResponse()
	if resp == nil {
		t.Fatal("expected a continue response")
	}

	mutation := resp.GetRequestBody().GetResponse().GetHeaderMutation()
	if mutation == nil {
		t.Fatal("skip continue response must carry a header mutation")
	}

	removeHeaders := mutation.GetRemoveHeaders()
	if !slices.Contains(removeHeaders, forwardedAuthorizationHeaderName) {
		t.Fatalf("skip processing must strip %q, got remove_headers=%v", forwardedAuthorizationHeaderName, removeHeaders)
	}
	if !slices.Contains(removeHeaders, headers.VSRInboundAuthorization) {
		t.Fatalf("skip processing must strip %q, got remove_headers=%v", headers.VSRInboundAuthorization, removeHeaders)
	}
	// The ext_authz-injected per-user provider keys must be stripped too, so a
	// header-injection auth deployment cannot leak them on the skip branch.
	for _, injected := range []string{headers.UserOpenAIKey, headers.UserAnthropicKey} {
		if !slices.Contains(removeHeaders, injected) {
			t.Fatalf("skip processing must strip injected key %q, got remove_headers=%v", injected, removeHeaders)
		}
	}
}

// TestSkipProcessingResponsePreservesCallerAuthorizationByDefault proves the
// strip is opt-in: with global.router.strip_inbound_authorization off (the
// default), the skip passthrough leaves the caller Authorization and the
// ext_authz-injected per-user keys in place, preserving pre-#2375 behavior. The
// internal looper carrier is still stripped unconditionally.
func TestSkipProcessingResponsePreservesCallerAuthorizationByDefault(t *testing.T) {
	cfg := &config.RouterConfig{} // StripInboundAuthorization defaults to false
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: newTestCredentialResolver(cfg),
	}

	resp := router.newContinueRequestBodyResponse()
	removeHeaders := resp.GetRequestBody().GetResponse().GetHeaderMutation().GetRemoveHeaders()

	if slices.Contains(removeHeaders, forwardedAuthorizationHeaderName) {
		t.Fatalf("default (opt-in off) must not strip %q, got remove_headers=%v", forwardedAuthorizationHeaderName, removeHeaders)
	}
	for _, injected := range []string{headers.UserOpenAIKey, headers.UserAnthropicKey} {
		if slices.Contains(removeHeaders, injected) {
			t.Fatalf("default (opt-in off) must not strip injected key %q, got remove_headers=%v", injected, removeHeaders)
		}
	}
	// The internal carrier is always stripped, independent of the opt-in.
	if !slices.Contains(removeHeaders, headers.VSRInboundAuthorization) {
		t.Fatalf("carrier %q must always be stripped, got remove_headers=%v", headers.VSRInboundAuthorization, removeHeaders)
	}
}

func TestHandleResponseHeadersSkipProcessingBypassesVSRHeaders(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SkipProcessing:          true,
		VSRSelectedCategory:     "math",
		VSRReasoningMode:        "on",
		VSRSelectedDecisionName: "math_decision",
		VSRSelectedModel:        "deepseek-v31",
	}
	responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
				},
			},
		},
	}

	response, err := router.handleResponseHeaders(responseHeaders, ctx)
	if err != nil {
		t.Fatalf("handleResponseHeaders failed: %v", err)
	}
	if response.GetResponseHeaders() == nil {
		t.Fatal("expected response headers continue response")
	}
	if response.GetResponseHeaders().Response.Status != ext_proc.CommonResponse_CONTINUE {
		t.Fatalf("expected CONTINUE status, got %v", response.GetResponseHeaders().Response.Status)
	}
	if response.GetResponseHeaders().Response.HeaderMutation != nil {
		t.Fatal("expected no header mutation when skipping processing")
	}
	if response.ModeOverride != nil {
		t.Fatal("did not expect mode override when skipping processing on a non-streaming response")
	}
}

func TestHandleResponseHeadersSkipProcessingPreservesStreamingMode(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{SkipProcessing: true}
	responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "text/event-stream"},
				},
			},
		},
	}

	response, err := router.handleResponseHeaders(responseHeaders, ctx)
	if err != nil {
		t.Fatalf("handleResponseHeaders failed: %v", err)
	}
	if !ctx.IsStreamingResponse {
		t.Fatal("expected streaming response detection to remain active for downstream filters")
	}
	if response.ModeOverride == nil {
		t.Fatal("expected streaming mode override to be propagated")
	}
	if response.ModeOverride.ResponseBodyMode != http_ext.ProcessingMode_STREAMED {
		t.Fatalf("unexpected response body mode: %v", response.ModeOverride.ResponseBodyMode)
	}
}

func TestHandleResponseBodySkipProcessingShortCircuits(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SkipProcessing: true,
		RequestModel:   "passthrough",
	}
	responseBody := &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{
			Body:        []byte(`{"choices":[{"message":{"content":"hi"}}]}`),
			EndOfStream: true,
		},
	}

	response, err := router.handleResponseBody(responseBody, ctx)
	if err != nil {
		t.Fatalf("handleResponseBody failed: %v", err)
	}
	if response.GetResponseBody() == nil {
		t.Fatalf("expected response body continue response, got %T", response.GetResponse())
	}
	if response.GetResponseBody().Response.Status != ext_proc.CommonResponse_CONTINUE {
		t.Fatalf("expected CONTINUE status, got %v", response.GetResponseBody().Response.Status)
	}
	if response.GetResponseBody().Response.GetBodyMutation() != nil {
		t.Fatal("expected no body mutation when skipping processing")
	}
	if response.GetResponseBody().Response.GetHeaderMutation() != nil {
		t.Fatal("expected no header mutation when skipping processing")
	}
}

func TestSkipProcessingHelpersReturnNilWhenNotOptedOut(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{}

	if resp := router.handleSkipProcessingResponseHeaders(nil, ctx); resp != nil {
		t.Fatalf("expected nil header response when SkipProcessing is false, got %v", resp)
	}
	if resp := router.handleSkipProcessingResponseBody(nil, ctx); resp != nil {
		t.Fatalf("expected nil body response when SkipProcessing is false, got %v", resp)
	}
	if resp := router.handleSkipProcessingResponseHeaders(nil, nil); resp != nil {
		t.Fatalf("expected nil header response on nil context, got %v", resp)
	}
	if resp := router.handleSkipProcessingResponseBody(nil, nil); resp != nil {
		t.Fatalf("expected nil body response on nil context, got %v", resp)
	}
}
