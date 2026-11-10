package extproc

import (
	"context"
	"net/http"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func benchmarkHeaderRequest(expected string) *ext_proc.ProcessingRequest_RequestHeaders {
	return &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
		{Key: ":method", Value: "POST"}, {Key: ":path", Value: "/v1/chat/completions"}, {Key: headers.VSRBenchExpectedConfigHash, Value: expected},
	}}}}
}

func TestBenchmarkConfigPreconditionRejectsBeforeGeneration(t *testing.T) {
	active := strings.Repeat("a", 64)
	for _, tc := range []struct {
		name, expected, actual string
		status                 int
	}{
		{"match", active, active, 0},
		{"mismatch", strings.Repeat("b", 64), active, http.StatusPreconditionFailed},
		{"unavailable", active, "", http.StatusServiceUnavailable},
		{"malformed", "bad", active, http.StatusBadRequest},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{DocumentHash: tc.actual}}
			ctx := &RequestContext{Headers: map[string]string{}, TraceContext: context.Background()}
			response, err := router.handleRequestHeaders(benchmarkHeaderRequest(tc.expected), ctx)
			if err != nil {
				t.Fatal(err)
			}
			if tc.status == 0 {
				if response.GetRequestHeaders() == nil {
					t.Fatal("matching request did not continue")
				}
			} else if got := int(response.GetImmediateResponse().GetStatus().GetCode()); got != tc.status {
				t.Fatalf("status=%d want=%d", got, tc.status)
			}
		})
	}
}

// The receipt names the generation the request was admitted on, in the trace
// of every response, refusals included, for a caller allowed it.
func TestBenchmarkReceiptNamesTheAdmittedGeneration(t *testing.T) {
	active := strings.Repeat("a", 64)
	router := &OpenAIRouter{Config: &config.RouterConfig{DocumentHash: active}}
	router.Config.Headers.Debug = config.DebugHeadersAny
	for _, expected := range []string{active, strings.Repeat("b", 64)} {
		ctx := &RequestContext{Headers: map[string]string{}, TraceContext: context.Background()}
		request := benchmarkHeaderRequest(expected)
		request.RequestHeaders.Headers.Headers = append(request.RequestHeaders.Headers.Headers, &core.HeaderValue{Key: headers.VSRDebug, Value: "true"})
		response, err := router.handleRequestHeaders(request, ctx)
		if err != nil {
			t.Fatal(err)
		}
		if ctx.BenchmarkConfigHash != active {
			t.Fatalf("admitted generation = %q, want the active %q", ctx.BenchmarkConfigHash, active)
		}
		if immediate := response.GetImmediateResponse(); immediate != nil {
			appendRoutedResponseHeaders(response, ctx)
			if got := traceOf(t, immediate.GetHeaders().GetSetHeaders())[headers.TraceConfigHash]; got != `"`+active+`"` {
				t.Fatalf("refusal receipt = %s, want the active generation", got)
			}
		}
	}
}

// Envoy neither retries nor hedges a config-bound evaluation request, so its
// receipt counts every provider call.
func TestBenchmarkBindingDisablesEnvoyRetries(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{DocumentHash: strings.Repeat("a", 64)}}
	ctx := &RequestContext{Headers: map[string]string{headers.VSRBenchExpectedConfigHash: strings.Repeat("a", 64)}}
	response := newContinueRequestHeadersResponse()
	router.bindBenchmarkConfigResponse(response, ctx)
	set := headerValuesByName(response.GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders())
	if set[headers.EnvoyMaxRetries] != "0" || set[headers.EnvoyHedgeOnPerTryTimeout] != "false" {
		t.Fatalf("binding set %v, want no retries and no hedging", set)
	}
}
