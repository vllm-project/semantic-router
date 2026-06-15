package responseapi

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTranslateToCompletionRequest_StringInput(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`"What is 2+2?"`),
	}

	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)

	assert.Equal(t, "gpt-4o", result.Model)
	require.Len(t, result.Messages, 1)
	assert.NotNil(t, result.Messages[0].OfUser)
	assert.Equal(t, "What is 2+2?", result.Messages[0].OfUser.Content.OfString.Value)
}

func TestTranslateToCompletionRequest_ArrayInput(t *testing.T) {
	tr := NewTranslator()
	input := []InputItem{
		{Type: "message", Role: "user", Content: json.RawMessage(`"Hello"`)},
		{Type: "message", Role: "assistant", Content: json.RawMessage(`"Hi there"`)},
		{Type: "message", Role: "user", Content: json.RawMessage(`"How are you?"`)},
	}
	inputJSON, err := json.Marshal(input)
	require.NoError(t, err)

	req := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: inputJSON,
	}

	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)

	require.Len(t, result.Messages, 3)
	assert.NotNil(t, result.Messages[0].OfUser)
	assert.NotNil(t, result.Messages[1].OfAssistant)
	assert.NotNil(t, result.Messages[2].OfUser)
}

func TestTranslateToCompletionRequest_WithInstructions(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model:        "gpt-4o",
		Input:        json.RawMessage(`"hi"`),
		Instructions: "Be brief.",
	}

	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)

	require.Len(t, result.Messages, 2)
	assert.NotNil(t, result.Messages[0].OfSystem)
	assert.Equal(t, "Be brief.", result.Messages[0].OfSystem.Content.OfString.Value)
	assert.NotNil(t, result.Messages[1].OfUser)
}

func TestTranslateToCompletionRequest_InheritedInstructions(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`"hi"`),
	}

	history := []*StoredResponse{{
		Instructions: "You are a math tutor.",
		Input:        []InputItem{},
		Output:       []OutputItem{},
	}}

	result, err := tr.TranslateToCompletionRequest(req, history)
	require.NoError(t, err)

	require.Len(t, result.Messages, 2)
	assert.NotNil(t, result.Messages[0].OfSystem)
	assert.Equal(t, "You are a math tutor.", result.Messages[0].OfSystem.Content.OfString.Value)
}

func TestTranslateToCompletionRequest_OptionalParams(t *testing.T) {
	tr := NewTranslator()
	temp := 0.7
	topP := 0.9
	maxTokens := 100
	req := &ResponseAPIRequest{
		Model:           "gpt-4o",
		Input:           json.RawMessage(`"hi"`),
		Temperature:     &temp,
		TopP:            &topP,
		MaxOutputTokens: &maxTokens,
	}

	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)

	assert.Equal(t, 0.7, result.Temperature.Value)
	assert.Equal(t, 0.9, result.TopP.Value)
	assert.Equal(t, int64(100), result.MaxTokens.Value)
}

func TestTranslateToCompletionRequest_WithFunctionTools(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`"What's the weather?"`),
		Tools: []Tool{
			{
				Type: ToolTypeFunction,
				Function: &FunctionDef{
					Name:        "get_weather",
					Description: "Get weather",
				},
			},
		},
	}

	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)

	require.Len(t, result.Tools, 1)
	assert.Equal(t, "get_weather", result.Tools[0].Function.Name)
}

func TestTranslateToCompletionRequest_ImageGenToolStripped(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`"Draw a cat"`),
		Tools: []Tool{
			{Type: ToolTypeImageGeneration},
		},
	}

	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)

	assert.Empty(t, result.Tools, "image_generation tools must be stripped")
}

func TestTranslateToCompletionRequest_EmptyInputRejected(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(``),
	}

	_, err := tr.TranslateToCompletionRequest(req, nil)
	assert.Error(t, err)
}

func TestTranslateToResponseAPIResponse_TextContent(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model:        "gpt-4o",
		Instructions: "Be helpful.",
	}
	resp := &openai.ChatCompletion{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1700000000,
		Model:   "gpt-4o",
		Choices: []openai.ChatCompletionChoice{{
			Index: 0,
			Message: openai.ChatCompletionMessage{
				Role:    "assistant",
				Content: "Four.",
			},
			FinishReason: "stop",
		}},
		Usage: openai.CompletionUsage{
			PromptTokens:     10,
			CompletionTokens: 1,
			TotalTokens:      11,
		},
	}

	result := tr.TranslateToResponseAPIResponse(req, resp, "")

	assert.Equal(t, "response", result.Object)
	assert.Equal(t, StatusCompleted, result.Status)
	assert.Equal(t, "gpt-4o", result.Model)
	assert.Equal(t, "Four.", result.OutputText)
	require.Len(t, result.Output, 1)
	assert.Equal(t, ItemTypeMessage, result.Output[0].Type)
	require.Len(t, result.Output[0].Content, 1)
	assert.Equal(t, ContentTypeOutputText, result.Output[0].Content[0].Type)
	assert.Equal(t, "Four.", result.Output[0].Content[0].Text)
	assert.Equal(t, "Be helpful.", result.Instructions)
}

func TestTranslateToResponseAPIResponse_ToolCalls(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{Model: "gpt-4o"}
	resp := &openai.ChatCompletion{
		Model: "gpt-4o",
		Choices: []openai.ChatCompletionChoice{{
			Message: openai.ChatCompletionMessage{
				Role: "assistant",
				ToolCalls: []openai.ChatCompletionMessageToolCall{{
					ID:   "call_abc",
					Type: "function",
					Function: openai.ChatCompletionMessageToolCallFunction{
						Name:      "get_weather",
						Arguments: `{"city":"NYC"}`,
					},
				}},
			},
			FinishReason: "tool_calls",
		}},
		Usage: openai.CompletionUsage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
	}

	result := tr.TranslateToResponseAPIResponse(req, resp, "prev_resp_id")

	assert.Equal(t, "prev_resp_id", result.PreviousResponseID)
	require.Len(t, result.Output, 1)
	assert.Equal(t, ItemTypeFunctionCall, result.Output[0].Type)
	assert.Equal(t, "call_abc", result.Output[0].CallID)
	assert.Equal(t, "get_weather", result.Output[0].Name)
	assert.JSONEq(t, `{"city":"NYC"}`, result.Output[0].Arguments)
}

func TestTranslateToResponseAPIResponse_UsageMapping(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{Model: "gpt-4o"}
	resp := &openai.ChatCompletion{
		Model: "gpt-4o",
		Choices: []openai.ChatCompletionChoice{{
			Message:      openai.ChatCompletionMessage{Role: "assistant", Content: "ok"},
			FinishReason: "stop",
		}},
		Usage: openai.CompletionUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}

	result := tr.TranslateToResponseAPIResponse(req, resp, "")

	require.NotNil(t, result.Usage)
	assert.Equal(t, 100, result.Usage.InputTokens)
	assert.Equal(t, 50, result.Usage.OutputTokens)
	assert.Equal(t, 150, result.Usage.TotalTokens)
}

func TestTranslateToResponseAPIResponse_ResponseIDFormat(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{Model: "gpt-4o"}
	resp := &openai.ChatCompletion{
		Model:   "gpt-4o",
		Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "ok"}, FinishReason: "stop"}},
	}

	result := tr.TranslateToResponseAPIResponse(req, resp, "")

	assert.NotEmpty(t, result.ID, "response ID must be set")
	assert.Contains(t, result.ID, "resp_", "response ID must have resp_ prefix")
}

func TestTranslateToResponseAPIResponse_OutputItemIDs(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{Model: "gpt-4o"}
	resp := &openai.ChatCompletion{
		Model: "gpt-4o",
		Choices: []openai.ChatCompletionChoice{{
			Message:      openai.ChatCompletionMessage{Role: "assistant", Content: "hello"},
			FinishReason: "stop",
		}},
	}

	result := tr.TranslateToResponseAPIResponse(req, resp, "")

	require.Len(t, result.Output, 1)
	assert.Contains(t, result.Output[0].ID, "item_", "output item ID must have item_ prefix")
}

func TestOutputItemToMessage_FunctionCall(t *testing.T) {
	tr := NewTranslator()
	item := OutputItem{
		Type:      ItemTypeFunctionCall,
		CallID:    "call_123",
		Name:      "search",
		Arguments: `{"q":"test"}`,
	}

	msg, err := tr.outputItemToMessage(item)
	require.NoError(t, err)

	assert.NotNil(t, msg.OfAssistant)
	require.Len(t, msg.OfAssistant.ToolCalls, 1)
	assert.Equal(t, "call_123", msg.OfAssistant.ToolCalls[0].ID)
	assert.Equal(t, "search", msg.OfAssistant.ToolCalls[0].Function.Name)
}

func TestOutputItemToMessage_FunctionCallOutput(t *testing.T) {
	tr := NewTranslator()
	item := OutputItem{
		Type:   ItemTypeFunctionCallOutput,
		CallID: "call_123",
		Output: `{"result":"42"}`,
	}

	msg, err := tr.outputItemToMessage(item)
	require.NoError(t, err)

	assert.NotNil(t, msg.OfTool)
	assert.Equal(t, "call_123", msg.OfTool.ToolCallID)
	assert.JSONEq(t, `{"result":"42"}`, msg.OfTool.Content.OfString.Value)
}

func TestOutputItemToMessage_UnknownType(t *testing.T) {
	tr := NewTranslator()
	item := OutputItem{Type: "unknown_type"}

	_, err := tr.outputItemToMessage(item)
	assert.Error(t, err)
}

func TestTranslateToCompletionRequest_HistoryConversation(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`"What about 3+3?"`),
	}

	history := []*StoredResponse{{
		Input: []InputItem{
			{Role: "user", Content: json.RawMessage(`"What is 2+2?"`)},
		},
		Output: []OutputItem{
			{
				Type: ItemTypeMessage,
				Role: "assistant",
				Content: []ContentPart{
					{Type: ContentTypeOutputText, Text: "4"},
				},
			},
		},
	}}

	result, err := tr.TranslateToCompletionRequest(req, history)
	require.NoError(t, err)

	require.Len(t, result.Messages, 3)
	assert.NotNil(t, result.Messages[0].OfUser)
	assert.Equal(t, "What is 2+2?", result.Messages[0].OfUser.Content.OfString.Value)
	assert.NotNil(t, result.Messages[1].OfAssistant)
	assert.Equal(t, "4", result.Messages[1].OfAssistant.Content.OfString.Value)
	assert.NotNil(t, result.Messages[2].OfUser)
	assert.Equal(t, "What about 3+3?", result.Messages[2].OfUser.Content.OfString.Value)
}

func TestRoundTrip_ResponseAPIRequestToJSON(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`"test"`),
	}

	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)

	body, err := json.Marshal(result)
	require.NoError(t, err)

	var reparsed openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal(body, &reparsed))

	assert.Equal(t, "gpt-4o", reparsed.Model)
	require.Len(t, reparsed.Messages, 1)
	assert.NotNil(t, reparsed.Messages[0].OfUser)
}

func TestConvertTools_MixedTypes(t *testing.T) {
	tr := NewTranslator()
	tools := []Tool{
		{Type: ToolTypeFunction, Function: &FunctionDef{Name: "fn1", Description: "A function"}},
		{Type: ToolTypeImageGeneration},
		{Type: "code_interpreter"},
		{Type: ToolTypeFunction, Function: &FunctionDef{Name: "fn2", Description: "A function"}},
		{Type: "mcp"},
	}

	result := tr.convertTools(tools)

	require.Len(t, result, 2)
	assert.Equal(t, "fn1", result[0].Function.Name)
	assert.Equal(t, "fn2", result[1].Function.Name)
}

func TestConvertToolChoice_StringAuto(t *testing.T) {
	result := convertToolChoice("auto")
	body, err := json.Marshal(result)
	require.NoError(t, err)
	assert.JSONEq(t, `"auto"`, string(body))
}

func TestConvertToolChoice_StringNone(t *testing.T) {
	result := convertToolChoice("none")
	body, err := json.Marshal(result)
	require.NoError(t, err)
	assert.JSONEq(t, `"none"`, string(body))
}

func TestConvertToolChoice_StringRequired(t *testing.T) {
	result := convertToolChoice("required")
	body, err := json.Marshal(result)
	require.NoError(t, err)
	assert.JSONEq(t, `"required"`, string(body))
}

func TestConvertToolChoice_NamedFunction(t *testing.T) {
	named := map[string]interface{}{
		"type":     "function",
		"function": map[string]interface{}{"name": "get_weather"},
	}
	result := convertToolChoice(named)
	body, err := json.Marshal(result)
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &parsed))
	assert.Equal(t, "function", parsed["type"])
	fn := parsed["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", fn["name"])
}

func TestConvertToolChoice_InvalidFallback(t *testing.T) {
	result := convertToolChoice(map[string]interface{}{"garbage": true})
	body, err := json.Marshal(result)
	require.NoError(t, err)
	assert.Equal(t, "null", string(body))
}

func TestTranslateToCompletionRequest_ToolChoiceForwarded(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`"Use the tool"`),
		Tools: []Tool{
			{Type: ToolTypeFunction, Function: &FunctionDef{Name: "calc", Description: "A calculator"}},
		},
		ToolChoice: "required",
	}
	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)

	body, err := json.Marshal(result.ToolChoice)
	require.NoError(t, err)
	assert.JSONEq(t, `"required"`, string(body))
}

func TestInputItemToMessage_MultimodalImageContent(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`[{
			"type": "message",
			"role": "user",
			"content": [
				{"type": "input_text", "text": "What is in this image?"},
				{"type": "input_image", "image_url": "https://example.com/cat.jpg"}
			]
		}]`),
	}
	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)

	msg := result.Messages[0]
	require.NotNil(t, msg.OfUser)
	require.NotNil(t, msg.OfUser.Content.OfArrayOfContentParts)
	require.Len(t, msg.OfUser.Content.OfArrayOfContentParts, 2)

	textPart := msg.OfUser.Content.OfArrayOfContentParts[0]
	require.NotNil(t, textPart.OfText)
	assert.Equal(t, "What is in this image?", textPart.OfText.Text)

	imgPart := msg.OfUser.Content.OfArrayOfContentParts[1]
	require.NotNil(t, imgPart.OfImageURL)
	assert.Equal(t, "https://example.com/cat.jpg", imgPart.OfImageURL.ImageURL.URL)
}

func TestInputItemToMessage_TextOnlyNoImagePartsFallback(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`[{
			"type": "message",
			"role": "user",
			"content": [
				{"type": "input_text", "text": "Just text, no images."}
			]
		}]`),
	}
	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)

	msg := result.Messages[0]
	require.NotNil(t, msg.OfUser)
	require.NotNil(t, msg.OfUser.Content.OfString)
	assert.Equal(t, "Just text, no images.", msg.OfUser.Content.OfString.Value)
}

func TestTranslateToResponseAPIResponse_ZeroUsage(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{Model: "gpt-4o", Input: json.RawMessage(`"test"`)}
	resp := &openai.ChatCompletion{
		ID:    "chatcmpl-zero",
		Model: "gpt-4o",
		Choices: []openai.ChatCompletionChoice{{
			Message: openai.ChatCompletionMessage{Content: "Reply", Role: "assistant"},
		}},
	}
	result := tr.TranslateToResponseAPIResponse(req, resp, "")

	require.NotNil(t, result.Usage)
	assert.Equal(t, 0, result.Usage.InputTokens)
	assert.Equal(t, 0, result.Usage.OutputTokens)
	assert.Equal(t, 0, result.Usage.TotalTokens)
}

func TestMultimodalImageContent_RoundTrip_JSON(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`[{
			"type": "message",
			"role": "user",
			"content": [
				{"type": "input_text", "text": "Describe this."},
				{"type": "input_image", "image_url": "https://example.com/img.png", "detail": "high"}
			]
		}]`),
	}
	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)

	body, err := json.Marshal(result)
	require.NoError(t, err)

	var reparsed map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &reparsed))

	msgs := reparsed["messages"].([]interface{})
	require.Len(t, msgs, 1)
	msg := msgs[0].(map[string]interface{})
	assert.Equal(t, "user", msg["role"])

	content := msg["content"].([]interface{})
	require.Len(t, content, 2)
	textPart := content[0].(map[string]interface{})
	assert.Equal(t, "text", textPart["type"])
	assert.Equal(t, "Describe this.", textPart["text"])

	imgPart := content[1].(map[string]interface{})
	assert.Equal(t, "image_url", imgPart["type"])
	imgURL := imgPart["image_url"].(map[string]interface{})
	assert.Equal(t, "https://example.com/img.png", imgURL["url"])
	assert.Equal(t, "high", imgURL["detail"])
}
