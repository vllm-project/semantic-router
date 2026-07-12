package extproc

import (
	"net/http"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

func TestHandleRequestHeadersAcceptsCompleteLooperEnvelope(t *testing.T) {
	authenticator, internalHeaders := newLooperAuthTestFixture(t)
	values := authenticatedLooperEnvelope(internalHeaders,
		&core.HeaderValue{Key: headers.VSRLooperDecision, Value: "decision-a"},
		&core.HeaderValue{Key: headers.VSRLooperIteration, Value: "2"},
		&core.HeaderValue{Key: headers.VSRFusionDepth, Value: "3"},
	)

	ctx, response := processLooperEnvelope(t, authenticator, values)
	if response.GetRequestHeaders() == nil {
		t.Fatal("complete authenticated envelope did not continue")
	}
	if !ctx.LooperRequest || ctx.LooperDecision != "decision-a" || ctx.LooperIteration != 2 {
		t.Fatalf("authenticated metadata = (%t, %q, %d), want (true, decision-a, 2)",
			ctx.LooperRequest, ctx.LooperDecision, ctx.LooperIteration)
	}
}

func TestHandleRequestHeadersRejectsMalformedLooperEnvelopes(t *testing.T) {
	authenticator, internalHeaders := newLooperAuthTestFixture(t)
	validRequest := internalHeaders.Get(headers.VSRLooperRequest)
	validSecret := internalHeaders.Get(headers.VSRLooperSecret)
	authenticated := func(extras ...*core.HeaderValue) []*core.HeaderValue {
		return append([]*core.HeaderValue{
			{Key: headers.VSRLooperRequest, Value: validRequest},
			{Key: headers.VSRLooperSecret, Value: validSecret},
		}, extras...)
	}

	tests := []struct {
		name     string
		headers  []*core.HeaderValue
		canaries []string
	}{
		{name: "zero iteration", headers: authenticated(&core.HeaderValue{Key: headers.VSRLooperIteration, Value: "0"})},
		{name: "negative iteration", headers: authenticated(&core.HeaderValue{Key: headers.VSRLooperIteration, Value: "-1"})},
		{name: "non numeric iteration", headers: authenticated(&core.HeaderValue{Key: headers.VSRLooperIteration, Value: "not-a-number"})},
		{name: "overflowing iteration", headers: authenticated(&core.HeaderValue{Key: headers.VSRLooperIteration, Value: "9223372036854775808"})},
		{name: "empty iteration", headers: authenticated(&core.HeaderValue{Key: headers.VSRLooperIteration, Value: " "})},
		{name: "zero fusion depth", headers: authenticated(&core.HeaderValue{Key: headers.VSRFusionDepth, Value: "0"})},
		{name: "invalid fusion depth", headers: authenticated(&core.HeaderValue{Key: headers.VSRFusionDepth, Value: "deep"})},
		{
			name: "duplicate marker",
			headers: authenticated(
				&core.HeaderValue{Key: headers.VSRLooperRequest, Value: validRequest},
			),
		},
		{
			name: "duplicate decision",
			headers: authenticated(
				&core.HeaderValue{Key: headers.VSRLooperDecision, Value: "decision-a"},
				&core.HeaderValue{Key: headers.VSRLooperDecision, Value: "decision-b"},
			),
			canaries: []string{"decision-a", "decision-b"},
		},
		{
			name: "duplicate iteration",
			headers: authenticated(
				&core.HeaderValue{Key: headers.VSRLooperIteration, Value: "1"},
				&core.HeaderValue{Key: headers.VSRLooperIteration, Value: "2"},
			),
		},
		{
			name: "secret only partial envelope",
			headers: []*core.HeaderValue{
				{Key: headers.VSRLooperSecret, Value: validSecret},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx, response := processLooperEnvelope(t, authenticator, test.headers)
			assertMalformedLooperEnvelopeRejected(t, ctx, response, append(test.canaries, validSecret))
		})
	}
}

func TestLooperAuthenticationFailureLogOmitsQuery(t *testing.T) {
	logCore, observed := observer.New(zapcore.WarnLevel)
	previousLogger := zap.L()
	zap.ReplaceGlobals(zap.New(logCore))
	t.Cleanup(func() { zap.ReplaceGlobals(previousLogger) })

	authenticator, internalHeaders := newLooperAuthTestFixture(t)
	processLooperEnvelope(t, authenticator, []*core.HeaderValue{
		{Key: headers.VSRLooperRequest, Value: internalHeaders.Get(headers.VSRLooperRequest)},
	})

	entries := observed.FilterMessage("looper_request_authentication_failed").All()
	if len(entries) != 1 {
		t.Fatalf("authentication failure log count = %d, want 1", len(entries))
	}
	fields := entries[0].ContextMap()
	if got := fields["path"]; got != "/v1/chat/completions" {
		t.Fatalf("logged path = %q, want normalized path without query", got)
	}
	if strings.Contains(entries[0].ContextMap()["path"].(string), "query-canary") {
		t.Fatal("authentication failure warning retained the query canary")
	}
}

func newLooperAuthTestFixture(t *testing.T) (*looper.RequestAuthenticator, http.Header) {
	t.Helper()
	authenticator, err := looper.NewRequestAuthenticator()
	if err != nil {
		t.Fatalf("NewRequestAuthenticator() error = %v", err)
	}
	internalHeaders := make(http.Header)
	authenticator.Apply(internalHeaders)
	return authenticator, internalHeaders
}

func authenticatedLooperEnvelope(
	internalHeaders http.Header,
	extras ...*core.HeaderValue,
) []*core.HeaderValue {
	return append([]*core.HeaderValue{
		{Key: headers.VSRLooperRequest, Value: internalHeaders.Get(headers.VSRLooperRequest)},
		{Key: headers.VSRLooperSecret, Value: internalHeaders.Get(headers.VSRLooperSecret)},
	}, extras...)
}

func processLooperEnvelope(
	t *testing.T,
	authenticator *looper.RequestAuthenticator,
	internalHeaders []*core.HeaderValue,
) (*RequestContext, *ext_proc.ProcessingResponse) {
	t.Helper()
	values := append([]*core.HeaderValue{
		{Key: ":method", Value: "POST"},
		{Key: ":path", Value: "/v1/chat/completions?sensitive=query-canary"},
		{Key: "x-request-id", Value: "request-id-canary"},
	}, internalHeaders...)
	ctx := &RequestContext{Headers: make(map[string]string)}
	response, err := (&OpenAIRouter{
		looperAuthenticator: authenticator,
	}).handleRequestHeaders(&ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: values},
		},
	}, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders() error = %v", err)
	}
	return ctx, response
}

func assertMalformedLooperEnvelopeRejected(
	t *testing.T,
	ctx *RequestContext,
	response *ext_proc.ProcessingResponse,
	canaries []string,
) {
	t.Helper()
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		t.Fatal("malformed envelope did not produce an immediate response")
	}
	if got := immediate.GetStatus().GetCode(); got != typev3.StatusCode_Forbidden {
		t.Fatalf("status = %s, want Forbidden", got)
	}
	body := string(immediate.GetBody())
	if !strings.Contains(body, "invalid internal request authentication") {
		t.Fatalf("body = %q, want generic authentication error", body)
	}
	for _, canary := range canaries {
		if canary != "" && strings.Contains(body, canary) {
			t.Fatalf("generic error exposed internal metadata canary %q", canary)
		}
	}
	if ctx.LooperRequest || ctx.LooperDecision != "" || ctx.LooperIteration != 0 {
		t.Fatalf("malformed envelope populated typed context: %#v", ctx)
	}
	for _, name := range looperInternalRequestHeaders {
		if _, ok := ctx.Headers[name]; ok {
			t.Fatalf("malformed internal header %q entered generic context", name)
		}
	}
}
