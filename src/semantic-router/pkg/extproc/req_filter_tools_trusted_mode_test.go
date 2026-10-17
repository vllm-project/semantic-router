package extproc

import (
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// trustedModeNoneAvailability lists the tools-database controls for mode none.
// Unavailable, never-loaded, and stale evidence narrow; a fresh load allows.
// Every control must emit the same request.
type trustedModeNoneAvailability struct {
	name string
	db   func(t *testing.T) *tools.ToolsDatabase
	age  time.Duration
	want llmprotocol.TrustedOutcome
}

func trustedModeNoneAvailabilityControls() []trustedModeNoneAvailability {
	return []trustedModeNoneAvailability{
		{name: "unavailable", db: func(*testing.T) *tools.ToolsDatabase { return nil }, want: llmprotocol.TrustedNarrow},
		{name: "unloaded", db: func(*testing.T) *tools.ToolsDatabase {
			return tools.NewToolsDatabase(tools.ToolsDatabaseOptions{Enabled: true})
		}, want: llmprotocol.TrustedNarrow},
		{name: "stale", db: loadedTrustedFactsToolsDB, age: 2 * time.Minute, want: llmprotocol.TrustedNarrow},
		{name: "loaded", db: loadedTrustedFactsToolsDB, want: llmprotocol.TrustedAllow},
	}
}

func (c trustedModeNoneAvailability) apply(t *testing.T) *tools.ToolsDatabase {
	t.Helper()
	trustedFactsNow = func() time.Time { return time.Now().Add(c.age) }
	t.Cleanup(func() { trustedFactsNow = time.Now })
	return c.db(t)
}

// trustedModeNoneConfig is a valid enabled tools plugin in mode none with an
// authoritative operator-policy grant for the candidate stage, so only
// availability evidence decides between narrow and allow.
func trustedModeNoneConfig(stripHistory bool) *config.ToolsPluginConfig {
	return &config.ToolsPluginConfig{
		Enabled:          true,
		Mode:             config.ToolsPluginModeNone,
		StripToolHistory: stripHistory,
		TrustedFacts: &config.TrustedFactsConfig{
			Enabled:          true,
			Enforcement:      config.TrustedEnforcementAuthoritative,
			TrustSources:     []string{config.TrustedSourceOperatorPolicy},
			FreshnessSeconds: 60,
			StageRoles:       []string{config.TrustedStageCandidate},
		},
	}
}

// trustedModeNoneBody is a tool-bearing request with one prior tool call and
// result, so the emitted request shows tools, tool_choice, and tool history.
func trustedModeNoneBody(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIResponsesV1:
		return []byte(`{"model":"public-model","store":false,"input":[` +
			`{"role":"user","content":"weather today?"},` +
			`{"type":"function_call","call_id":"call_1","name":"search","arguments":"{}"},` +
			`{"type":"function_call_output","call_id":"call_1","output":"sunny"},` +
			`{"role":"user","content":"and tomorrow?"}],` +
			`"tools":[{"type":"function","name":"search","parameters":{"type":"object"}},` +
			`{"type":"function","name":"calculator","parameters":{"type":"object"}}],"tool_choice":"auto"}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"model":"public-model","max_tokens":64,"messages":[` +
			`{"role":"user","content":"weather today?"},` +
			`{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"search","input":{}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"sunny"},{"type":"text","text":"and tomorrow?"}]}],` +
			`"tools":[{"name":"search","input_schema":{"type":"object"}},{"name":"calculator","input_schema":{"type":"object"}}],` +
			`"tool_choice":{"type":"auto"}}`)
	default:
		return []byte(`{"model":"public-model","messages":[` +
			`{"role":"user","content":"weather today?"},` +
			`{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{}"}}]},` +
			`{"role":"tool","tool_call_id":"call_1","content":"sunny"},` +
			`{"role":"user","content":"and tomorrow?"}],` +
			`"tools":[{"type":"function","function":{"name":"search","parameters":{"type":"object"}}},` +
			`{"type":"function","function":{"name":"calculator","parameters":{"type":"object"}}}],"tool_choice":"auto"}`)
	}
}

// requireModeNoneWire decodes the provider-bound body and checks that mode none
// held: no tools, no tool_choice, and tool history only when it is retained.
func requireModeNoneWire(t *testing.T, format llmprotocol.WireFormat, body []byte, stripHistory bool) {
	t.Helper()
	decoded, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(format, body)
	require.NoError(t, err, "emitted body: %s", body)
	require.Empty(t, decoded.Tools, "mode none must emit no tools: %s", body)
	require.Empty(t, decoded.ToolChoice.Mode, "mode none must emit no tool_choice: %s", body)
	facts := extractSemanticRequestSignals(&decoded)
	wantHistory := 1
	if stripHistory {
		wantHistory = 0
	}
	require.Equal(t, wantHistory, facts.AssistantToolCallCount, "tool calls in %s", body)
	require.Equal(t, wantHistory, facts.ToolResultCount, "tool results in %s", body)
}

func pendingTrustedFactsVerdicts(ctx *RequestContext) []string {
	verdicts := make([]string, 0, len(ctx.pendingTrustedFactsOutcomes))
	for _, outcome := range ctx.pendingTrustedFactsOutcomes {
		verdicts = append(verdicts, outcome.Verdict)
	}
	return verdicts
}

// Narrow returns before ordinary mode handling, so it must apply mode none
// itself. Each availability control must emit what the loaded control emits.
func TestHandleToolSelectionTrustedFactsNarrowKeepsModeNone(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1,
	} {
		for _, control := range trustedModeNoneAvailabilityControls() {
			for _, strip := range []bool{false, true} {
				name := string(format) + "/" + control.name + "/keep_history"
				if strip {
					name = string(format) + "/" + control.name + "/strip_history"
				}
				t.Run(name, func(t *testing.T) {
					router := &OpenAIRouter{
						ToolsDatabase:     control.apply(t),
						ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore()),
					}
					ctx := trustedFactsTestContext(t, trustedModeNoneConfig(strip))
					ctx.SourceFormat = format
					ctx.TraceContext = t.Context()
					req, immediate := router.prepareProtocolRequest(trustedModeNoneBody(format), ctx)
					require.Nil(t, immediate)
					require.Len(t, req.Tools, 2, "fixture must carry client tools")

					router.handleToolSelectionForRequest(req, &ext_proc.ProcessingResponse{}, ctx)

					require.Equal(t, []string{string(control.want)}, pendingTrustedFactsVerdicts(ctx))
					engine, err := router.protocolEngine()
					require.NoError(t, err)
					encoded, err := engine.EncodeRequest(format, *ctx.SemanticRequest, ctx.ProtocolEnvelope)
					require.NoError(t, err)
					requireModeNoneWire(t, format, encoded.Body, strip)
				})
			}
		}
	}
}

// Through the full routing path, every availability control dispatches the
// same mode-none body for each client protocol.
func TestModelRoutingTrustedFactsModeNoneAcrossAvailability(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1,
	} {
		for _, control := range trustedModeNoneAvailabilityControls() {
			t.Run(string(format)+"/"+control.name, func(t *testing.T) {
				router, model := routingTestRouterForFormat(format)
				router.ToolsDatabase = control.apply(t)
				router.ResponseAPIFilter = NewResponseAPIFilter(NewMockResponseStore())
				ctx := &RequestContext{SourceFormat: format, TraceContext: t.Context(), Headers: map[string]string{}}
				request, immediate := router.prepareProtocolRequest(trustedModeNoneBody(format), ctx)
				require.Nil(t, immediate)
				ctx.VSRSelectedDecision = &config.Decision{
					Name:    "trusted-mode-none",
					Plugins: []config.DecisionPlugin{mustToolsDecisionPlugin(t, trustedModeNoneConfig(true))},
				}

				response, err := router.handleModelRouting(request, model, "trusted-mode-none", entropy.ReasoningDecision{}, model, ctx)
				require.NoError(t, err)
				require.NotNil(t, response.GetRequestBody())
				requireModeNoneWire(t, format, response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), true)
			})
		}
	}
}
