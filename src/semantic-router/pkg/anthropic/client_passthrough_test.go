package anthropic

import (
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
