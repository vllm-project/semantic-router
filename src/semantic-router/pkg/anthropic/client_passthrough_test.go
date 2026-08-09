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

func TestReplay_ImageBlocksReachedOutboundUserMessage_Base64(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("what is this")},
	}
	pt := &AnthropicPassthrough{
		UserMessageImageBlocks: map[int][]ImageBlock{
			0: {{Source: ImageSource{Type: "base64", MediaType: "image/png", Data: "AAAA"}}},
		},
	}

	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)

	content := gjson.GetBytes(body, "messages.0.content").Array()
	require.Len(t, content, 2)
	assert.Equal(t, "text", content[0].Get("type").String())
	assert.Equal(t, "image", content[1].Get("type").String())
	assert.Equal(t, "base64", content[1].Get("source.type").String())
	assert.Equal(t, "image/png", content[1].Get("source.media_type").String())
	assert.Equal(t, "AAAA", content[1].Get("source.data").String())
}

func TestReplay_ImageBlocksReachedOutboundUserMessage_URL(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("describe this")},
	}
	pt := &AnthropicPassthrough{
		UserMessageImageBlocks: map[int][]ImageBlock{
			0: {{Source: ImageSource{Type: "url", URL: "https://example.com/cat.png"}}},
		},
	}

	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)
	content := gjson.GetBytes(body, "messages.0.content").Array()
	require.Len(t, content, 2)
	assert.Equal(t, "url", content[1].Get("source.type").String())
	assert.Equal(t, "https://example.com/cat.png", content[1].Get("source.url").String())
}

func TestReplay_ImageBlocksMapToUserMessageIndexOnly(t *testing.T) {
	// Assistant messages are skipped when counting user-message indices.
	req := &openai.ChatCompletionNewParams{
		Model: "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{
			userStringMessage("first"),
			assistantStringMessage("ack"),
			userStringMessage("second"),
		},
	}
	pt := &AnthropicPassthrough{
		UserMessageImageBlocks: map[int][]ImageBlock{
			1: {{Source: ImageSource{Type: "url", URL: "https://example.com/b.png"}}},
		},
	}
	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)
	// Image should land on messages[2] (user index 1), not messages[1] (assistant).
	assert.Len(t, gjson.GetBytes(body, "messages.0.content").Array(), 1)
	assert.Len(t, gjson.GetBytes(body, "messages.1.content").Array(), 1)
	assert.Len(t, gjson.GetBytes(body, "messages.2.content").Array(), 2)
	assert.Equal(t, "image", gjson.GetBytes(body, "messages.2.content.1.type").String())
}

func TestReplay_ImageBlocksDroppedWhenIndexOutOfRange(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model:    "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{userStringMessage("hi")},
	}
	pt := &AnthropicPassthrough{
		UserMessageImageBlocks: map[int][]ImageBlock{
			5: {{Source: ImageSource{Type: "url", URL: "https://example.com/a.png"}}},
		},
	}
	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)
	assert.Len(t, gjson.GetBytes(body, "messages.0.content").Array(), 1)
}

// OpenAI-shape image_url inbound -> Anthropic image block.
// Closes the OpenAI-client/Anthropic-backend cell for image content. No
// passthrough required for this path: the OpenAI message itself carries the
// image part.
func TestOpenAIShape_ImageURLDataURIBecomesAnthropicImageBlock(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model: "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfArrayOfContentParts: []openai.ChatCompletionContentPartUnionParam{
						{OfText: &openai.ChatCompletionContentPartTextParam{Text: "what is this"}},
						{OfImageURL: &openai.ChatCompletionContentPartImageParam{
							ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
								URL: "data:image/png;base64,AAAA",
							},
						}},
					},
				},
			}},
		},
	}

	body, err := ToAnthropicRequestBody(req)
	require.NoError(t, err)
	content := gjson.GetBytes(body, "messages.0.content").Array()
	require.Len(t, content, 2)
	assert.Equal(t, "text", content[0].Get("type").String())
	assert.Equal(t, "what is this", content[0].Get("text").String())
	assert.Equal(t, "image", content[1].Get("type").String())
	assert.Equal(t, "base64", content[1].Get("source.type").String())
	assert.Equal(t, "image/png", content[1].Get("source.media_type").String())
	assert.Equal(t, "AAAA", content[1].Get("source.data").String())
}

func TestOpenAIShape_ImageURLPlainURLBecomesAnthropicURLImage(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Model: "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfArrayOfContentParts: []openai.ChatCompletionContentPartUnionParam{
						{OfImageURL: &openai.ChatCompletionContentPartImageParam{
							ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
								URL: "https://example.com/cat.png",
							},
						}},
					},
				},
			}},
		},
	}
	body, err := ToAnthropicRequestBody(req)
	require.NoError(t, err)
	content := gjson.GetBytes(body, "messages.0.content").Array()
	require.Len(t, content, 1)
	assert.Equal(t, "image", content[0].Get("type").String())
	assert.Equal(t, "url", content[0].Get("source.type").String())
	assert.Equal(t, "https://example.com/cat.png", content[0].Get("source.url").String())
}

func TestBuildRequestHeadersWithPassthrough_NilByteIdenticalToLegacy(t *testing.T) {
	legacy := BuildRequestHeaders("k", 100, "")
	pt := BuildRequestHeadersWithPassthrough("k", 100, "", nil)
	assert.Equal(t, legacy, pt)
}

func TestBuildRequestHeadersWithPassthrough_VersionOverridesDefault(t *testing.T) {
	headers := BuildRequestHeadersWithPassthrough("k", 1, "", &AnthropicPassthrough{
		AnthropicVersion: "2024-10-22",
	})
	got := headerMap(headers)
	assert.Equal(t, "2024-10-22", got["anthropic-version"])
}

func TestBuildRequestHeadersWithPassthrough_VersionFallsBackToDefault(t *testing.T) {
	headers := BuildRequestHeadersWithPassthrough("k", 1, "", &AnthropicPassthrough{})
	got := headerMap(headers)
	assert.Equal(t, AnthropicAPIVersion, got["anthropic-version"])
}

func TestBuildRequestHeadersWithPassthrough_BetaForwarded(t *testing.T) {
	headers := BuildRequestHeadersWithPassthrough("k", 1, "", &AnthropicPassthrough{
		AnthropicBeta: "prompt-caching-2024-07-31",
	})
	got := headerMap(headers)
	assert.Equal(t, "prompt-caching-2024-07-31", got["anthropic-beta"])
}

func TestBuildRequestHeadersWithPassthrough_BetaOmittedWhenEmpty(t *testing.T) {
	headers := BuildRequestHeadersWithPassthrough("k", 1, "", &AnthropicPassthrough{})
	got := headerMap(headers)
	_, ok := got["anthropic-beta"]
	assert.False(t, ok)
}

func TestBuildStreamingRequestHeadersWithPassthrough_PreservesAcceptAndBeta(t *testing.T) {
	headers := BuildStreamingRequestHeadersWithPassthrough("k", 1, "", &AnthropicPassthrough{
		AnthropicBeta: "messages-2023-12-15",
	})
	got := headerMap(headers)
	assert.Equal(t, "text/event-stream", got["accept"])
	assert.Equal(t, "messages-2023-12-15", got["anthropic-beta"])
}

// withToolResultRequest builds a request whose third message is an OpenAI
// tool message that the writer translates into an Anthropic tool_result block.
func withToolResultRequest() *openai.ChatCompletionNewParams {
	return &openai.ChatCompletionNewParams{
		Model: "claude-sonnet-4-5",
		Messages: []openai.ChatCompletionMessageParamUnion{
			userStringMessage("weather?"),
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
					OfString: openai.String("backend error"),
				},
			}},
		},
	}
}

func TestReplay_ToolResultErrorPreserved(t *testing.T) {
	req := withToolResultRequest()
	pt := &AnthropicPassthrough{
		ToolResultErrors: map[string]bool{"call_abc": true},
	}

	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)

	// The tool_result lives on messages[2] (the user-wrapped tool message).
	isErr := gjson.GetBytes(body, "messages.2.content.0.is_error")
	require.True(t, isErr.Exists())
	assert.True(t, isErr.Bool())
}

func TestReplay_ToolResultErrorDefaultsToFalse(t *testing.T) {
	// Today's translation always emits is_error (SDK NewToolResultBlock sets
	// it via param.Bool). With no passthrough flag set, the value remains the
	// default false; the passthrough only flips it to true when present.
	req := withToolResultRequest()
	body, err := ToAnthropicRequestBodyWithPassthrough(req, &AnthropicPassthrough{})
	require.NoError(t, err)
	isErr := gjson.GetBytes(body, "messages.2.content.0.is_error")
	require.True(t, isErr.Exists())
	assert.False(t, isErr.Bool())
}

func TestReplay_ToolResultArrayContent_TextAndImageMix(t *testing.T) {
	req := withToolResultRequest()
	pt := &AnthropicPassthrough{
		ToolResultArrayContent: map[string][]ToolResultContentBlock{
			"call_abc": {
				{Type: "text", Text: "see image"},
				{Type: "image", Source: &ImageSource{
					Type: "base64", MediaType: "image/jpeg", Data: "BBBB",
				}},
			},
		},
	}

	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)

	content := gjson.GetBytes(body, "messages.2.content.0.content").Array()
	require.Len(t, content, 2)
	assert.Equal(t, "text", content[0].Get("type").String())
	assert.Equal(t, "see image", content[0].Get("text").String())
	assert.Equal(t, "image", content[1].Get("type").String())
	assert.Equal(t, "base64", content[1].Get("source.type").String())
	assert.Equal(t, "image/jpeg", content[1].Get("source.media_type").String())
	assert.Equal(t, "BBBB", content[1].Get("source.data").String())
}

func TestReplay_ToolResultUnknownIDLeavesBlockUntouched(t *testing.T) {
	// An ID in the passthrough that doesn't match an outbound tool_result
	// must not affect the existing block's is_error value (defaults to false).
	req := withToolResultRequest()
	pt := &AnthropicPassthrough{
		ToolResultErrors: map[string]bool{"call_other": true},
	}
	body, err := ToAnthropicRequestBodyWithPassthrough(req, pt)
	require.NoError(t, err)
	isErr := gjson.GetBytes(body, "messages.2.content.0.is_error")
	require.True(t, isErr.Exists())
	assert.False(t, isErr.Bool())
}

func headerMap(headers []HeaderKeyValue) map[string]string {
	out := make(map[string]string, len(headers))
	for _, h := range headers {
		out[h.Key] = h.Value
	}
	return out
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
