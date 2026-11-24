package extproc

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func unservedModelsRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.5
providers:
  defaults:
    model: a
  models:
    - name: a
      backend_refs: [{endpoint: 127.0.0.1:8000, protocol: http, provider: vllm}]
recipes:
  - name: default
`))
	require.NoError(t, err)
	return &OpenAIRouter{Config: cfg}
}

// With a route, a model the listener does not accept or the Router does not
// serve is routed as the route's model and kept as the requested one; an
// accepted model, an empty one and a request without a route stay as named.
func TestRouteUnservedModel(t *testing.T) {
	router := unservedModelsRouter(t)
	for _, test := range []struct {
		name, model, route, want, requested string
		allowed                             routing.ListenerModels
	}{
		{name: "a client's own model", model: "gpt-5.6-sol", route: "vllm-sr/auto", want: "vllm-sr/auto", requested: "gpt-5.6-sol"},
		{name: "a provider model the listener does not list", model: "a", route: "vllm-sr/auto", allowed: routing.ListenerModels{"vllm-sr/auto"}, want: "vllm-sr/auto", requested: "a"},
		{name: "a provider model the listener accepts", model: "a", route: "vllm-sr/auto", want: "a"},
		{name: "the route itself", model: "vllm-sr/auto", route: "vllm-sr/auto", allowed: routing.ListenerModels{"vllm-sr/auto"}, want: "vllm-sr/auto"},
		{name: "no route", model: "gpt-5.6-sol", want: "gpt-5.6-sol"},
		{name: "no model", model: "", route: "vllm-sr/auto", want: ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := &RequestContext{UnservedModelRoute: test.route, ListenerModels: test.allowed, SemanticRequest: &llmprotocol.Request{Model: test.model}}
			require.Equal(t, test.want, router.routeUnservedModel(test.model, ctx))
			require.Equal(t, test.requested, ctx.RequestedModel)
			if test.requested != "" {
				require.Equal(t, test.route, ctx.SemanticRequest.Model)
			}
		})
	}

	ctx := &RequestContext{UnservedModelRoute: "vllm-sr/auto"}
	router.routeUnservedModel(strings.Repeat("m", 300), ctx)
	require.Len(t, ctx.RequestedModel, maxRequestedModelBytes)
}

// The trace names the model the client asked for when the listener routed it.
func TestTraceNamesTheRequestedModel(t *testing.T) {
	ctx := &RequestContext{RequestedModel: "gpt-5.6-sol", SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1}
	trace := headers.Trace{}
	addRoutingTrace(&trace, ctx)
	require.Contains(t, trace.Encode(), `requested_model="gpt-5.6-sol"`)

	plain := headers.Trace{}
	addRoutingTrace(&plain, &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1})
	require.NotContains(t, plain.Encode(), "requested_model")
}

// The Replay record keeps the requested model beside the model it was
// routed as.
func TestReplayRecordKeepsTheRequestedModel(t *testing.T) {
	ctx := &RequestContext{RequestedModel: "gpt-5.6-sol", Headers: map[string]string{}}
	record := buildReplayRoutingRecord(ctx, "vllm-sr/auto", "a", "answer")
	require.Equal(t, "gpt-5.6-sol", record.RequestedModel)
	require.Equal(t, "vllm-sr/auto", record.OriginalModel)
}
