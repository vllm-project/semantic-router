package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Chat stop and Messages stop_sequences set stop sequences, and they count
// only when the request states no reasoning preference beside them. Responses
// has no stop sequences: the member is refused.
func TestUnstatedReasoningStopSequencesPerFormat(t *testing.T) {
	const messages = `"messages":[{"role":"user","content":"What is 17*23?"}]`
	for _, test := range []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
		want   bool
	}{
		{"messages stop_sequences", llmprotocol.AnthropicMessagesV1, `{"model":"m","max_tokens":1024,"stop_sequences":["17*23"],` + messages + `}`, true},
		{"messages stop_sequences with adaptive thinking", llmprotocol.AnthropicMessagesV1, `{"model":"m","max_tokens":1024,"stop_sequences":["17*23"],"thinking":{"type":"adaptive"},` + messages + `}`, false},
		{"messages without stop_sequences", llmprotocol.AnthropicMessagesV1, `{"model":"m","max_tokens":1024,` + messages + `}`, false},
		{"chat stop string", llmprotocol.OpenAIChatV1, `{"model":"m","stop":"17*23",` + messages + `}`, true},
		{"chat stop list", llmprotocol.OpenAIChatV1, `{"model":"m","stop":["\n\n","17*23"],` + messages + `}`, true},
		{"chat empty stop list", llmprotocol.OpenAIChatV1, `{"model":"m","stop":[],` + messages + `}`, false},
		{"chat stop with reasoning_effort", llmprotocol.OpenAIChatV1, `{"model":"m","stop":"17*23","reasoning_effort":"low",` + messages + `}`, false},
		{"chat stop with a thinking switch", llmprotocol.OpenAIChatV1, `{"model":"m","stop":"17*23","chat_template_kwargs":{"enable_thinking":true},` + messages + `}`, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(test.format, []byte(test.body))
			require.NoError(t, err)
			require.Equal(t, test.want, unstatedReasoningStopSequences(request))
		})
	}
	_, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1,
		[]byte(`{"model":"m","stop":["17*23"],"input":"What is 17*23?"}`))
	require.Error(t, err, "a Responses request with stop")
}

// A fallback candidate runs with the reasoning its lane chose unless the
// request sets stop sequences without a reasoning preference, as the primary
// does. A decision that blocks stop keeps it.
func TestFallbackCandidateReasoningFollowsStopSequences(t *testing.T) {
	router := reasoningBudgetRouter(t, llmprotocol.OpenAIChatV1, "qwen3", 0)
	blocksStop := &config.Decision{Name: "thinking", Plugins: []config.DecisionPlugin{{
		Type:          config.DecisionPluginRequestParams,
		Configuration: config.MustStructuredPayload(map[string]any{"blocked_params": []string{"stop"}}),
	}}}
	for _, test := range []struct {
		name     string
		stops    bool
		decision *config.Decision
		want     bool
	}{
		{"stop sequences", true, nil, false},
		{"no stop sequences, or a preference", false, nil, true},
		{"stop sequences the decision blocks", true, blocksStop, true},
	} {
		ctx := &RequestContext{
			VSREligibleModelRefs:           []config.ModelRef{{Model: "worker", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPtr(true)}}},
			VSRSelectedDecision:            test.decision,
			unstatedReasoningStopSequences: test.stops,
		}
		require.Equal(t, test.want, router.candidateReasoningChoice(ctx, "worker"), test.name)
	}
}

const blockedStop = `        plugins:
          - type: request_params
            configuration:
              blocked_params: [stop]`

// vLLM matches stop sequences against the reasoning as well as the answer, so
// a decision lane that turns reasoning on for a client that asked for none
// ends inside the reasoning whenever the reasoning writes a stop sequence. Such
// a request reaches the backend with the model's reasoning-off control and its
// stop sequence, and the trace and Router Replay say so. An explicit
// preference, a family without an off control, or a decision that blocks stop
// keeps the reasoning the decision chose.
func TestStopSequencesTurnTheDecisionsReasoningOff(t *testing.T) {
	const question = `"messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}]`
	messages, chat := llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1
	stopsOff := "off;reason=stop_sequences"
	for _, test := range []struct {
		name      string
		client    llmprotocol.WireFormat
		backend   llmprotocol.WireFormat
		family    string
		threshold int
		plugins   string
		body      string
		want      string
		trace     string
		forwarded bool
	}{
		{
			"messages stop_sequences without thinking", messages, chat, "qwen3", 0, "",
			`{"model":"vllm-sr/auto","max_tokens":1024,"stop_sequences":["17*23"],` + question + `}`,
			"enable_thinking=false", stopsOff, true,
		},
		{
			"messages without stop_sequences", messages, chat, "qwen3", 0, "",
			`{"model":"vllm-sr/auto","max_tokens":1024,` + question + `}`,
			"enable_thinking=true", "on", false,
		},
		{
			"messages stop_sequences with adaptive thinking", messages, chat, "qwen3", 0, "",
			`{"model":"vllm-sr/auto","max_tokens":1024,"thinking":{"type":"adaptive"},"stop_sequences":["17*23"],` + question + `}`,
			"enable_thinking=true", "on", true,
		},
		{
			"messages stop_sequences with thinking disabled", messages, chat, "qwen3", 0, "",
			`{"model":"vllm-sr/auto","max_tokens":1024,"thinking":{"type":"disabled"},"stop_sequences":["17*23"],` + question + `}`,
			"enable_thinking=false", "off", true,
		},
		{
			"messages stop_sequences to a messages backend", messages, messages, "claude-effort-adaptive", 0, "",
			`{"model":"vllm-sr/auto","max_tokens":1024,"stop_sequences":["17*23"],` + question + `}`,
			"disabled", stopsOff, true,
		},
		{
			"chat stop without reasoning", chat, chat, "qwen3", 0, "",
			`{"model":"vllm-sr/auto","stop":"17*23",` + question + `}`,
			"enable_thinking=false", stopsOff, true,
		},
		{
			"chat stop with reasoning_effort", chat, chat, "qwen3", 0, "",
			`{"model":"vllm-sr/auto","stop":"17*23","reasoning_effort":"low",` + question + `}`,
			"enable_thinking=true", "on", true,
		},
		{
			"chat stop with a thinking switch", chat, chat, "qwen3", 0, "",
			`{"model":"vllm-sr/auto","stop":["17*23"],"chat_template_kwargs":{"enable_thinking":true},` + question + `}`,
			"enable_thinking=true", "on", true,
		},
		{
			"chat stop to a family that cannot turn reasoning off", chat, chat, "qwen3-always-on", 0, "",
			`{"model":"vllm-sr/auto","stop":"17*23",` + question + `}`,
			"enable_thinking=true", "on", true,
		},
		{
			"chat stop the decision blocks", chat, chat, "qwen3", 0, blockedStop,
			`{"model":"vllm-sr/auto","stop":"17*23",` + question + `}`,
			"enable_thinking=true", "on", false,
		},
		{
			"chat stop and a small output limit", chat, chat, "qwen3", 512, "",
			`{"model":"vllm-sr/auto","max_tokens":100,"stop":"17*23",` + question + `}`,
			"enable_thinking=false", "off;reason=output_budget", true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			router := reasoningLaneRouter(t, test.backend, test.family, test.threshold, test.plugins)
			sent, ctx := routeReasoningRequest(t, router, test.client, test.body)
			require.Equal(t, test.want, sentReasoning(t, test.backend, sent), "backend body: %v", sent)
			var stops []string
			if test.forwarded {
				stops = []string{"17*23"}
			}
			require.Equal(t, stops, sentStops(t, test.backend, sent), "backend body: %v", sent)
			require.Equal(t, test.trace, traceReasoning(t, router, ctx))
			mode, _, _ := strings.Cut(test.trace, ";")
			require.Equal(t, mode, buildReplayRoutingRecord(ctx, "vllm-sr/auto", "worker", "thinking").ReasoningMode)
		})
	}
}

// sentStops reads the stop sequences the backend received: Chat stop, a
// string or a list, or Messages stop_sequences.
func sentStops(t *testing.T, backend llmprotocol.WireFormat, sent map[string]json.RawMessage) []string {
	t.Helper()
	member := "stop"
	if backend == llmprotocol.AnthropicMessagesV1 {
		member = "stop_sequences"
	}
	raw, found := sent[member]
	if !found {
		return nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return []string{one}
	}
	var many []string
	require.NoError(t, json.Unmarshal(raw, &many))
	return many
}
