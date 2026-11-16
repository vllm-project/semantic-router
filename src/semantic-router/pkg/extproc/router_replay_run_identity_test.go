package extproc

import (
	"context"
	"encoding/json"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// codexRunHeaders are the identity headers Codex 0.166.0 sends on thread of
// the run session, as captured live. A spawned agent also names its parent
// thread and its subagent kind.
func codexRunHeaders(session, thread, parent string) map[string]string {
	metadata, _ := json.Marshal(map[string]string{"session_id": session, "thread_id": thread, "turn_id": thread + "-turn"})
	values := map[string]string{"session-id": session, "thread-id": thread, "x-codex-turn-metadata": string(metadata)}
	if parent != "" {
		values["x-codex-parent-thread-id"] = parent
		values["x-openai-subagent"] = "collab_spawn"
	}
	return values
}

// recordRunCall starts the replay record of one harness call, a Responses
// call standing alone as Codex sends it with store false, and returns it.
func recordRunCall(t *testing.T, router *OpenAIRouter, requestHeaders map[string]string, format llmprotocol.WireFormat) store.Record {
	t.Helper()
	ctx := &RequestContext{SourceFormat: format, Headers: requestHeaders, RequestID: "call", TraceContext: t.Context()}
	body := []byte(`{"model":"vllm-sr/auto","max_tokens":8,"messages":[{"role":"user","content":"fix the build"}]}`)
	if format == llmprotocol.OpenAIResponsesV1 {
		body = []byte(`{"model":"vllm-sr/auto","store":false,"input":"fix the build"}`)
	}
	_, immediate := router.prepareProtocolRequest(body, ctx)
	require.Nil(t, immediate)
	if format == llmprotocol.OpenAIResponsesV1 {
		ctx.ResponseObjectState = &ResponseObjectState{SessionTrackingID: sessionTrackingStandalonePrefix + "resp_" + requestHeaders["thread-id"]}
	}
	ctx.RouterReplayPluginConfig = router.Config.EffectiveRouterReplayConfig(nil)
	router.startRouterReplay(ctx, "vllm-sr/auto", "qwen", "all")
	require.NotEmpty(t, ctx.RouterReplayID)
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.True(t, ok)
	return record
}

// A run is the session the harness names and the agent tree in it. Every
// call of a Codex run is filed under its session-id, whichever headers
// protection reads and whether protection can key the call, rather than one
// respapi:standalone session per call. Each call names its agent, the agent
// that spawned it and the harness, from what the harness sends.
func TestReplayFilesEachCallUnderItsHarnessRun(t *testing.T) {
	claudeHeaders := func(agent string) map[string]string {
		values := map[string]string{"x-claude-code-session-id": "cc-run"}
		if agent != "" {
			values["x-claude-code-agent-id"] = agent
		}
		return values
	}
	for _, tc := range []struct {
		name     string
		identity config.RouterLearningIdentityHeadersConfig
	}{
		{name: "default identity"},
		{name: "claude code identity headers", identity: config.RouterLearningIdentityHeadersConfig{
			Session: stringPtr("x-claude-code-session-id"), Agent: stringPtr("x-claude-code-agent-id"),
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}},
				replayTestPlugin(map[string]interface{}{"enabled": true}))
			cfg.RouterLearning.Protection.Identity.Headers = tc.identity
			router := replayRuntimeRouter(t, cfg)

			root := recordRunCall(t, router, codexRunHeaders("cx-run", "cx-run", ""), llmprotocol.OpenAIResponsesV1)
			child := recordRunCall(t, router, codexRunHeaders("cx-run", "child", "cx-run"), llmprotocol.OpenAIResponsesV1)
			grandchild := recordRunCall(t, router, codexRunHeaders("cx-run", "grandchild", "child"), llmprotocol.OpenAIResponsesV1)
			for _, record := range []store.Record{root, child, grandchild} {
				require.Equal(t, "cx-run", record.SessionID)
			}
			require.Equal(t, &store.AgentIdentity{Harness: store.HarnessCodex}, root.Agent)
			require.Equal(t, &store.AgentIdentity{
				Harness: store.HarnessCodex, ID: shortLearningIdentityHash("child"), Kind: "collab_spawn",
			}, child.Agent)
			require.Equal(t, &store.AgentIdentity{
				Harness: store.HarnessCodex, ID: shortLearningIdentityHash("grandchild"),
				ParentID: shortLearningIdentityHash("child"), Kind: "collab_spawn",
			}, grandchild.Agent)

			main := recordRunCall(t, router, claudeHeaders(""), llmprotocol.AnthropicMessagesV1)
			subagent := recordRunCall(t, router, claudeHeaders("a1b2"), llmprotocol.AnthropicMessagesV1)
			require.Equal(t, []string{"cc-run", "cc-run"}, []string{main.SessionID, subagent.SessionID})
			require.Equal(t, &store.AgentIdentity{Harness: store.HarnessClaudeCode}, main.Agent)
			require.Equal(t, &store.AgentIdentity{Harness: store.HarnessClaudeCode, ID: shortLearningIdentityHash("a1b2")}, subagent.Agent)
		})
	}
}

// A Looper hop carries none of its client's headers. Its record is filed in
// the run of the request that made it, under that request's agent, and names
// the request, rather than under a session the hop's own body derives.
func TestReplayFilesALooperHopInItsRequestsRun(t *testing.T) {
	cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}},
		replayTestPlugin(map[string]interface{}{"enabled": true}))
	router := replayRuntimeRouter(t, cfg)
	body := []byte(`{"model":"vllm-sr/auto","max_tokens":8,"messages":[{"role":"user","content":"fix the build"}]}`)
	client := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1, RequestID: "client-request", TraceContext: t.Context(),
		Headers: map[string]string{"x-claude-code-session-id": "cc-run", "x-claude-code-agent-id": "a1b2"},
	}
	_, immediate := router.prepareProtocolRequest(body, client)
	require.Nil(t, immediate)

	hopContext := routing.WithHop(router.withReplayRunParent(t.Context(), client), routing.Hop{Decision: "all"})
	hop := router.newRoutingSession(hopContext, nil).ctx
	hop.SourceFormat, hop.RequestID = llmprotocol.OpenAIChatV1, "hop-request"
	_, immediate = router.prepareProtocolRequest([]byte(`{"model":"qwen","messages":[{"role":"user","content":"fix the build"}]}`), hop)
	require.Nil(t, immediate)
	hop.RouterReplayPluginConfig = router.Config.EffectiveRouterReplayConfig(nil)
	router.startLooperInternalReplay(hop, "qwen", "all")
	record, ok := router.ReplayRecorder.GetRecord(hop.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, "cc-run", record.SessionID)
	require.Equal(t, "client-request", record.ParentRequestID)
	require.Equal(t, &store.AgentIdentity{Harness: store.HarnessClaudeCode, ID: shortLearningIdentityHash("a1b2")}, record.Agent)
}

// A harness that sends no trace context still gets one trace per run: each
// call's request span is parented under the run's agent span, so the main
// agent's calls and a subagent's share a trace id. A call that carries a
// traceparent stays in the harness's trace.
func TestRequestSpansJoinTheirRunsTrace(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder), sdktrace.WithIDGenerator(tracing.RunIDGenerator()))
	previous, previousPropagator := otel.GetTracerProvider(), otel.GetTextMapPropagator()
	otel.SetTracerProvider(provider)
	otel.SetTextMapPropagator(tracing.DefaultPropagator())
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previous)
		otel.SetTextMapPropagator(previousPropagator)
	})
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	request := func(extra map[string]string) trace.SpanContext {
		headers := newRequestHeaders("POST", "/v1/messages")
		for name, value := range extra {
			headers.RequestHeaders.Headers.Headers = append(headers.RequestHeaders.Headers.Headers, &core.HeaderValue{Key: name, Value: value})
		}
		ctx := &RequestContext{Headers: map[string]string{}, TraceContext: t.Context()}
		_, err := router.handleRequestHeaders(headers, ctx)
		require.NoError(t, err)
		require.NotNil(t, ctx.RequestSpan)
		ctx.RequestSpan.End()
		return ctx.RequestSpan.SpanContext()
	}
	main := request(map[string]string{"x-claude-code-session-id": "trace-run"})
	subagent := request(map[string]string{"x-claude-code-session-id": "trace-run", "x-claude-code-agent-id": "a1b2"})
	other := request(map[string]string{"x-claude-code-session-id": "another-run"})
	require.Equal(t, main.TraceID(), subagent.TraceID())
	require.NotEqual(t, main.TraceID(), other.TraceID())
	var subagentSpan sdktrace.ReadOnlySpan
	for _, span := range recorder.Ended() {
		if span.SpanContext().SpanID() == subagent.SpanID() {
			subagentSpan = span
		}
	}
	require.NotNil(t, subagentSpan)
	require.Contains(t, subagentSpan.Attributes(), attribute.String(tracing.AttrAgentID, shortLearningIdentityHash("a1b2")))

	harness := request(map[string]string{
		"x-claude-code-session-id": "trace-run",
		"traceparent":              "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
	})
	require.Equal(t, "0af7651916cd43dd8448eb211c80319c", harness.TraceID().String())
}
