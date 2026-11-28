package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestAddSystemPromptUsesRouterConfigBeforeGlobal(t *testing.T) {
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				decisionWithSystemPrompt("support", "global prompt"),
			},
		},
	})
	defer restoreGlobalConfig()

	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{
					decisionWithSystemPrompt("support", "router prompt"),
				},
			},
		},
	}

	ctx := &RequestContext{VSRSelectedDecision: &router.Config.Decisions[0]}
	body, err := router.addSystemPromptIfConfigured(
		[]byte(`{"messages":[{"role":"user","content":"hello"}]}`),
		"support",
		"test-model",
		ctx,
	)

	require.NoError(t, err)
	assert.Contains(t, string(body), "router prompt")
	assert.NotContains(t, string(body), "global prompt")
}

func TestAddSystemPromptDoesNotUseGlobalWhenRouterConfigMissesDecision(t *testing.T) {
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				decisionWithSystemPrompt("support", "global prompt"),
			},
		},
	})
	defer restoreGlobalConfig()

	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
	}
	originalBody := []byte(`{"messages":[{"role":"user","content":"hello"}]}`)

	body, err := router.addSystemPromptIfConfigured(
		originalBody,
		"support",
		"test-model",
		&RequestContext{},
	)

	require.NoError(t, err)
	assert.JSONEq(t, string(originalBody), string(body))
}

func TestAddSystemPromptMissingDecisionDoesNotPanic(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
	}
	originalBody := []byte(`{"messages":[{"role":"user","content":"hello"}]}`)

	body, err := router.addSystemPromptIfConfigured(
		originalBody,
		"missing",
		"test-model",
		&RequestContext{},
	)

	require.NoError(t, err)
	assert.JSONEq(t, string(originalBody), string(body))
}

func TestAddSystemPromptPreservesRawProviderFieldsAndIntegerPrecision(t *testing.T) {
	originalBody := []byte(`{
		"model":"provider/model",
		"opaque_integer":9007199254740993,
		"messages":[
			{"role":"system","content":[{"type":"text","text":"Existing instruction."}],"cache_control":{"type":"ephemeral"},"opaque_message_integer":9007199254740995},
			{"role":"assistant","content":"Prior answer.","reasoning_content":"opaque reasoning"},
			{"role":"user","content":"Continue."}
		]
	}`)

	body, injected, err := addSystemPromptToRequestBody(
		originalBody,
		"Treat inputs as confidential.",
		"insert",
	)
	require.NoError(t, err)
	require.True(t, injected)

	var decoded map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &decoded))
	assert.Equal(t, "9007199254740993", string(decoded["opaque_integer"]))
	var messages []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(decoded["messages"], &messages))
	require.Len(t, messages, 3)
	assert.JSONEq(t, `"Treat inputs as confidential.\n\nExisting instruction."`, string(messages[0]["content"]))
	assert.JSONEq(t, `{"type":"ephemeral"}`, string(messages[0]["cache_control"]))
	assert.JSONEq(t, "9007199254740995", string(messages[0]["opaque_message_integer"]))
	assert.JSONEq(t, `"opaque reasoning"`, string(messages[1]["reasoning_content"]))
}

// replace takes the place of every instruction the caller sent: each system
// and developer message, leading or later in the conversation, and the
// Responses instructions, whichever client protocol and backend.
func TestSystemPromptReplaceDropsEveryCallerInstruction(t *testing.T) {
	const messagesInConversation = `{"model":"m","max_tokens":16,"system":"CLIENT_TOP","messages":[` +
		`{"role":"user","content":"hi"},{"role":"assistant","content":"yo"},` +
		`{"role":"system","content":"CLIENT_LATE"},{"role":"user","content":"again"}]}`
	const laterTurns = `{"role":"system","content":"CLIENT_TOP"},{"role":"user","content":"hi"},{"role":"assistant","content":"yo"},` +
		`{"role":"developer","content":"CLIENT_LATE"},{"role":"user","content":"again"}`
	for _, tc := range []struct {
		name           string
		client, target llmprotocol.WireFormat
		body           string
	}{
		{
			"second system message", llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
			`{"model":"m","messages":[{"role":"system","content":"CLIENT_SYS_A"},{"role":"system","content":"CLIENT_SYS_B"},{"role":"user","content":"hi"}]}`,
		},
		{
			"developer before system", llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
			`{"model":"m","messages":[{"role":"developer","content":"CLIENT_DEV"},{"role":"system","content":"CLIENT_SYS"},{"role":"user","content":"hi"}]}`,
		},
		{
			"responses instructions", llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1,
			`{"model":"m","instructions":"CLIENT_INSTR","input":"hi"}`,
		},
		{
			"anthropic backend", llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
			`{"model":"m","max_tokens":16,"messages":[{"role":"system","content":"CLIENT_SYS"},{"role":"developer","content":"CLIENT_INSTR"},{"role":"user","content":"hi"}]}`,
		},
		{"messages client to chat", llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, messagesInConversation},
		{"messages client to responses", llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, messagesInConversation},
		{"messages client to messages", llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, messagesInConversation},
		{"later chat message", llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, `{"model":"m","messages":[` + laterTurns + `]}`},
		{"later responses item", llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1, `{"model":"m","max_output_tokens":16,"input":[` + laterTurns + `]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := dispatchWithReplacedSystemPrompt(t, tc.client, tc.target, tc.body)

			assert.Equal(t, 1, strings.Count(string(body), "ROUTER_PROMPT"), "%s", body)
			assert.NotContains(t, string(body), "CLIENT_")
		})
	}
}

// A system message that also sets per-message effort loses its text under
// replace but keeps the effort control in its place, so a Messages backend
// still receives it.
func TestSystemPromptReplaceKeepsPerMessageEffort(t *testing.T) {
	body := dispatchWithReplacedSystemPrompt(t, llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"},`+
			`{"role":"system","content":"CLIENT_LATE","output_config":{"effort":"low"}},{"role":"user","content":"again"}]}`)

	var dispatched struct {
		System   json.RawMessage `json:"system"`
		Messages []struct {
			Role         string          `json:"role"`
			Content      json.RawMessage `json:"content"`
			OutputConfig json.RawMessage `json:"output_config"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(body, &dispatched))
	assert.Contains(t, string(dispatched.System), "ROUTER_PROMPT")
	assert.NotContains(t, string(body), "CLIENT_")
	require.Len(t, dispatched.Messages, 4, "%s", body)
	assert.Equal(t, "system", dispatched.Messages[2].Role)
	assert.JSONEq(t, `[]`, string(dispatched.Messages[2].Content))
	assert.JSONEq(t, `{"effort":"low"}`, string(dispatched.Messages[2].OutputConfig))
}

func dispatchWithReplacedSystemPrompt(t *testing.T, client, target llmprotocol.WireFormat, body string) []byte {
	t.Helper()
	router, model := routingTestRouterForFormat(target)
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(client, []byte(body))
	require.NoError(t, err)
	ctx := routingTestContext(client, &request)
	decision := config.Decision{Name: "support", Plugins: []config.DecisionPlugin{{
		Type:          config.DecisionPluginSystemPrompt,
		Configuration: config.MustStructuredPayload(config.SystemPromptPluginConfig{SystemPrompt: "ROUTER_PROMPT", Mode: config.SystemPromptModeReplace}),
	}}}
	ctx.VSRSelectedDecision = &decision

	_, err = router.prepareProviderDispatch(&request, model, decision.Name, false, ctx)
	require.NoError(t, err)
	dispatched, err := router.encodeDispatchRequest(ctx)
	require.NoError(t, err)
	return dispatched
}

// replace drops what the caller sent, not the context the context plugin
// already wrote into the system prompt: the dispatched system message is the
// Router's prompt followed by each source's context.
func TestSystemPromptReplaceKeepsInjectedContext(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIChatV1,
		[]byte(`{"model":"m","messages":[{"role":"system","content":"CLIENT_SYS"},{"role":"user","content":"hi"}]}`))
	require.NoError(t, err)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, &request)
	contextPlugin := config.ContextPluginConfig{Enabled: true, Injection: &config.ContextInjectionConfig{Mode: config.ContextInjectionSystemPrompt}}
	decision := config.Decision{Name: "support", Plugins: []config.DecisionPlugin{
		{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(contextPlugin)},
		{Type: config.DecisionPluginSystemPrompt, Configuration: config.MustStructuredPayload(config.SystemPromptPluginConfig{SystemPrompt: "ROUTER_PROMPT", Mode: config.SystemPromptModeReplace})},
	}}
	ctx.VSRSelectedDecision = &decision
	require.NoError(t, router.injectContext(ctx, &contextPlugin, config.ContextSourceRAG, "RETRIEVED_DOC"))
	require.NoError(t, router.injectContext(ctx, &contextPlugin, config.ContextSourceMemory, "REMEMBERED_FACT"))

	_, err = router.prepareProviderDispatch(&request, model, decision.Name, false, ctx)
	require.NoError(t, err)
	body, err := router.encodeDispatchRequest(ctx)
	require.NoError(t, err)

	var dispatched struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(body, &dispatched))
	require.Len(t, dispatched.Messages, 2, "%s", body)
	assert.Equal(t, "system", dispatched.Messages[0].Role)
	system := string(dispatched.Messages[0].Content)
	prompt, retrieved, remembered := strings.Index(system, "ROUTER_PROMPT"), strings.Index(system, "RETRIEVED_DOC"), strings.Index(system, "REMEMBERED_FACT")
	assert.True(t, prompt >= 0 && prompt < retrieved && retrieved < remembered, "system message %s", system)
	assert.NotContains(t, string(body), "CLIENT_SYS")
}

func decisionWithSystemPrompt(name string, prompt string) config.Decision {
	return config.Decision{
		Name: name,
		Plugins: []config.DecisionPlugin{
			{
				Type: config.DecisionPluginSystemPrompt,
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled":       true,
					"system_prompt": prompt,
					"mode":          "insert",
				}),
			},
		},
	}
}
