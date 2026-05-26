package anthropic

import (
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

type byteEqFixture struct {
	name string
	req  *openai.ChatCompletionNewParams
}

func fixtureBasicUserOnly() byteEqFixture {
	return byteEqFixture{
		name: "basic_user_only",
		req: &openai.ChatCompletionNewParams{
			Model: "claude-sonnet-4-5",
			Messages: []openai.ChatCompletionMessageParamUnion{
				userStringMessage("Hello, world!"),
			},
		},
	}
}

func fixtureWithSystemPrompt() byteEqFixture {
	return byteEqFixture{
		name: "with_system_prompt",
		req: &openai.ChatCompletionNewParams{
			Model: "claude-sonnet-4-5",
			Messages: []openai.ChatCompletionMessageParamUnion{
				systemStringMessage("You are helpful."),
				userStringMessage("Hi"),
			},
		},
	}
}

func fixtureWithOptionalSamplingParams() byteEqFixture {
	return byteEqFixture{
		name: "with_optional_sampling_params",
		req: &openai.ChatCompletionNewParams{
			Model:       "claude-sonnet-4-5",
			Temperature: openai.Float(0.7),
			TopP:        openai.Float(0.9),
			MaxTokens:   openai.Int(1024),
			Stop: openai.ChatCompletionNewParamsStopUnion{
				OfStringArray: []string{"END", "STOP"},
			},
			Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("Hello")},
		},
	}
}

func fixtureMultiTurn() byteEqFixture {
	return byteEqFixture{
		name: "multi_turn",
		req: &openai.ChatCompletionNewParams{
			Model: "claude-sonnet-4-5",
			Messages: []openai.ChatCompletionMessageParamUnion{
				userStringMessage("What is 2+2?"),
				assistantStringMessage("2+2 equals 4."),
				userStringMessage("And 3+3?"),
			},
		},
	}
}

func fixtureWithToolHistory() byteEqFixture {
	return byteEqFixture{
		name: "with_tool_history",
		req: &openai.ChatCompletionNewParams{
			Model: "claude-sonnet-4-5",
			Messages: []openai.ChatCompletionMessageParamUnion{
				userStringMessage("Weather in Paris?"),
				{OfAssistant: &openai.ChatCompletionAssistantMessageParam{
					ToolCalls: []openai.ChatCompletionMessageToolCallParam{{
						ID:   "call_abc",
						Type: "function",
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      "get_weather",
							Arguments: `{"city":"Paris"}`,
						},
					}},
				}},
				{OfTool: &openai.ChatCompletionToolMessageParam{
					ToolCallID: "call_abc",
					Content: openai.ChatCompletionToolMessageParamContentUnion{
						OfString: openai.String(`{"temp_c": 18}`),
					},
				}},
			},
		},
	}
}

func userStringMessage(s string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{OfUser: &openai.ChatCompletionUserMessageParam{
		Content: openai.ChatCompletionUserMessageParamContentUnion{OfString: openai.String(s)},
	}}
}

func systemStringMessage(s string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{OfSystem: &openai.ChatCompletionSystemMessageParam{
		Content: openai.ChatCompletionSystemMessageParamContentUnion{OfString: openai.String(s)},
	}}
}

func assistantStringMessage(s string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{OfAssistant: &openai.ChatCompletionAssistantMessageParam{
		Content: openai.ChatCompletionAssistantMessageParamContentUnion{OfString: openai.String(s)},
	}}
}

// fixturesForByteEqualityCheck mirrors the request shapes exercised in
// client_test.go and compat_test.go, so the legacy entrypoint is checked
// against the passthrough-aware entrypoint with nil over the full surface
// today's tests cover.
func fixturesForByteEqualityCheck() []byteEqFixture {
	return []byteEqFixture{
		fixtureBasicUserOnly(),
		fixtureWithSystemPrompt(),
		fixtureWithOptionalSamplingParams(),
		fixtureMultiTurn(),
		fixtureWithToolHistory(),
	}
}

// TestToAnthropicRequestBodyWithPassthrough_NilIsByteIdenticalToLegacy is the
// safety net that lets subsequent commits replay Anthropic-only fields without
// risking regression for callers that don't pass a passthrough.
func TestToAnthropicRequestBodyWithPassthrough_NilIsByteIdenticalToLegacy(t *testing.T) {
	for _, fx := range fixturesForByteEqualityCheck() {
		t.Run(fx.name, func(t *testing.T) {
			legacy, err := ToAnthropicRequestBody(fx.req)
			require.NoError(t, err)

			withPT, err := ToAnthropicRequestBodyWithPassthrough(fx.req, nil)
			require.NoError(t, err)

			assert.Equal(t, string(legacy), string(withPT),
				"passthrough-aware entrypoint must be byte-identical when pt is nil")
		})
	}
}

func TestReplay_TopK(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("hi")},
	}
	k := int64(40)
	pt := &AnthropicPassthrough{TopK: &k}

	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)
	assert.Equal(t, int64(40), gjson.GetBytes(body, "top_k").Int())
}

func TestReplay_TopK_OmittedWhenNil(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("hi")},
	}
	body, err := ToAnthropicRequestBodyWithPassthrough(req, &AnthropicPassthrough{})
	require.NoError(t, err)
	assert.False(t, gjson.GetBytes(body, "top_k").Exists())
}

func TestReplay_MetadataUserID(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("hi")},
	}
	pt := &AnthropicPassthrough{MetadataUserID: "alice"}

	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)
	assert.Equal(t, "alice", gjson.GetBytes(body, "metadata.user_id").String())
}

func TestReplay_MetadataPrefersPassthroughOverOpenAIUser(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		User:     openai.String("openai-user"),
		Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("hi")},
	}
	pt := &AnthropicPassthrough{MetadataUserID: "passthrough-user"}

	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)
	assert.Equal(t, "passthrough-user", gjson.GetBytes(body, "metadata.user_id").String())
}

func TestReplay_MetadataFallsBackToOpenAIUser(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		User:     openai.String("openai-user"),
		Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("hi")},
	}
	body, err := ToAnthropicRequestBodyWithPassthrough(req, &AnthropicPassthrough{})
	require.NoError(t, err)
	assert.Equal(t, "openai-user", gjson.GetBytes(body, "metadata.user_id").String())
}

func TestReplay_SystemArrayWithCacheControl(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("hi")},
	}
	pt := &AnthropicPassthrough{
		SystemBlocks: []SystemBlock{
			{Text: "You are helpful.", CacheControl: &CacheControlSpec{Type: "ephemeral", TTL: "5m"}},
			{Text: "Be concise."},
		},
	}

	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)

	system := gjson.GetBytes(body, "system")
	require.True(t, system.IsArray())
	systemArr := system.Array()
	require.Len(t, systemArr, 2)
	assert.Equal(t, "You are helpful.", systemArr[0].Get("text").String())
	assert.Equal(t, "ephemeral", systemArr[0].Get("cache_control.type").String())
	assert.Equal(t, "5m", systemArr[0].Get("cache_control.ttl").String())
	assert.Equal(t, "Be concise.", systemArr[1].Get("text").String())
	assert.False(t, systemArr[1].Get("cache_control").Exists())
}

func TestReplay_SystemArrayOverridesStringFormSystem(t *testing.T) {
	// An OpenAI-derived system string is overridden by passthrough SystemBlocks.
	req := &openai.ChatCompletionNewParams{
		Model: "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{
			systemStringMessage("derived from OpenAI"),
			userStringMessage("hi"),
		},
	}
	pt := &AnthropicPassthrough{
		SystemBlocks: []SystemBlock{{Text: "from passthrough"}},
	}

	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)
	systemArr := gjson.GetBytes(body, "system").Array()
	require.Len(t, systemArr, 1)
	assert.Equal(t, "from passthrough", systemArr[0].Get("text").String())
}

func TestReplay_PerMessageCacheControl_OnExistingTextBlock(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("hello")},
	}
	pt := &AnthropicPassthrough{
		CacheControl: map[string]CacheControlSpec{
			"messages[0].content[0]": {Type: "ephemeral", TTL: "1h"},
		},
	}

	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)
	cc := gjson.GetBytes(body, "messages.0.content.0.cache_control")
	require.True(t, cc.Exists())
	assert.Equal(t, "ephemeral", cc.Get("type").String())
	assert.Equal(t, "1h", cc.Get("ttl").String())
}

func TestReplay_CacheControlDroppedWhenBlockMissing(t *testing.T) {
	// A marker for messages[5].content[0] when only one message exists must
	// not error; the marker is silently dropped.
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("hello")},
	}
	pt := &AnthropicPassthrough{
		CacheControl: map[string]CacheControlSpec{
			"messages[5].content[0]": {Type: "ephemeral"},
		},
	}
	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)
	assert.False(t, gjson.GetBytes(body, "messages.0.content.0.cache_control").Exists())
}

func TestReplay_ToolCacheControl(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("hi")},
		Tools: []openai.ChatCompletionToolParam{{
			Type: "function",
			Function: openai.FunctionDefinitionParam{
				Name: "get_weather",
				Parameters: openai.FunctionParameters{
					"type": "object",
				},
			},
		}},
	}
	pt := &AnthropicPassthrough{
		CacheControl: map[string]CacheControlSpec{
			"tools[0]": {Type: "ephemeral", TTL: "5m"},
		},
	}

	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)
	cc := gjson.GetBytes(body, "tools.0.cache_control")
	require.True(t, cc.Exists())
	assert.Equal(t, "ephemeral", cc.Get("type").String())
	assert.Equal(t, "5m", cc.Get("ttl").String())
}
