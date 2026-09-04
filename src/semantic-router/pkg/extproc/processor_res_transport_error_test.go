package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestUpstreamTransportErrorMalformedBodyUsesNativeClientFallback(t *testing.T) {
	router := &OpenAIRouter{}
	engine := protocolcodec.NewBuiltinEngine()
	for _, target := range extProcMatrixFormats {
		t.Run(string(target), func(t *testing.T) {
			ctx := &RequestContext{
				SourceFormat: target, TargetFormat: llmprotocol.OpenAIResponsesV1,
				UpstreamStatusCode: 503, RequestID: "request_1", RequestModel: "public-model",
			}
			response := router.handleUpstreamTransportError([]byte("upstream unavailable"), ctx)
			body := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
			translated, err := engine.TranslateTransportError(target, target, body, nil)
			if err != nil {
				t.Fatalf("fallback is not a valid %s error: %v\n%s", target, err, body)
			}
			protocolError := translated.TransportError.Error
			if protocolError == nil || protocolError.Category != llmprotocol.ErrorUpstreamUnavailable ||
				protocolError.Message != "model service returned an invalid error response" {
				t.Fatalf("fallback semantics = %+v", protocolError)
			}
		})
	}
}

func TestUpstreamTransportErrorMalformedBodyRetainsRateLimitSemantics(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
		UpstreamStatusCode: 429, RequestID: "request_1", RequestModel: "public-model",
	}
	response := router.handleUpstreamTransportError([]byte("not-json"), ctx)
	body := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
	translated, err := protocolcodec.NewBuiltinEngine().TranslateTransportError(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		body,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if translated.TransportError.Error == nil ||
		translated.TransportError.Error.Category != llmprotocol.ErrorRateLimited {
		t.Fatalf("rate-limit fallback semantics = %+v", translated.TransportError.Error)
	}
}

func TestUpstreamTransportFallbackPreservesOnlySafeStatusSemantics(t *testing.T) {
	tests := []struct {
		status   int
		category llmprotocol.ErrorCategory
		code     string
	}{
		{status: 400, category: llmprotocol.ErrorUpstreamUnavailable, code: "invalid_upstream_error"},
		{status: 401, category: llmprotocol.ErrorUpstreamUnavailable, code: "invalid_upstream_error"},
		{status: 408, category: llmprotocol.ErrorUpstreamTimeout, code: "upstream_timeout"},
		{status: 429, category: llmprotocol.ErrorRateLimited, code: "rate_limited"},
		{status: 500, category: llmprotocol.ErrorUpstreamUnavailable, code: "invalid_upstream_error"},
		{status: 504, category: llmprotocol.ErrorUpstreamTimeout, code: "upstream_timeout"},
		{status: 599, category: llmprotocol.ErrorUpstreamUnavailable, code: "invalid_upstream_error"},
	}
	for _, test := range tests {
		protocolError := upstreamTransportFallback(test.status, nil)
		if protocolError.Category != test.category || protocolError.Code != test.code ||
			protocolError.Message != "model service returned an invalid error response" {
			t.Fatalf("status %d fallback = %+v, want category=%q code=%q", test.status, protocolError, test.category, test.code)
		}
	}
}

func TestUpstreamTransportErrorDetectionRequiresObservedNon2xxStatus(t *testing.T) {
	for _, test := range []struct {
		status int
		want   bool
	}{
		{status: 0, want: false},
		{status: 199, want: true},
		{status: 200, want: false},
		{status: 299, want: false},
		{status: 300, want: true},
		{status: 429, want: true},
		{status: 503, want: true},
	} {
		if got := isUpstreamTransportError(&RequestContext{UpstreamStatusCode: test.status}); got != test.want {
			t.Fatalf("status %d: got %v, want %v", test.status, got, test.want)
		}
	}
	if isUpstreamTransportError(nil) {
		t.Fatal("nil context cannot have an observed transport error")
	}
}

func TestResponseWireFormatsDefaultsAndOrientation(t *testing.T) {
	chat := llmprotocol.OpenAIChatV1
	for _, test := range []struct {
		name       string
		ctx        *RequestContext
		wantSource llmprotocol.WireFormat
		wantTarget llmprotocol.WireFormat
	}{
		{name: "nil", wantSource: chat, wantTarget: chat},
		{name: "empty", ctx: &RequestContext{}, wantSource: chat, wantTarget: chat},
		{
			name:       "same-source-fallback",
			ctx:        &RequestContext{SourceFormat: llmprotocol.AnthropicMessagesV1},
			wantSource: llmprotocol.AnthropicMessagesV1, wantTarget: llmprotocol.AnthropicMessagesV1,
		},
		{
			name:       "cross-protocol",
			ctx:        &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.AnthropicMessagesV1},
			wantSource: llmprotocol.AnthropicMessagesV1, wantTarget: llmprotocol.OpenAIResponsesV1,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			source, target := responseWireFormats(test.ctx)
			if source != test.wantSource || target != test.wantTarget {
				t.Fatalf("formats = %q -> %q, want %q -> %q", source, target, test.wantSource, test.wantTarget)
			}
		})
	}
}
