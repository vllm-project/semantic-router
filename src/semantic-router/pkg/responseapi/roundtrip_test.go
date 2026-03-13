package responseapi

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockChatCompletionResponse returns a valid OpenAI Chat Completions JSON
// response body as if it came from an upstream vLLM/OpenAI backend.
func mockChatCompletionResponse(content, model string) []byte {
	resp := map[string]interface{}{
		"id":      "chatcmpl-mock-001",
		"object":  "chat.completion",
		"created": 1700000000,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": content,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     15,
			"completion_tokens": 8,
			"total_tokens":      23,
		},
	}
	body, _ := json.Marshal(resp)
	return body
}

func buildRoundTripFixtures(t *testing.T) (*Translator, *ResponseAPIRequest, *httptest.Server) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(mockChatCompletionResponse("Paris is the capital of France.", "gpt-4o"))
	}))

	return NewTranslator(), &ResponseAPIRequest{
		Model:        "gpt-4o",
		Input:        json.RawMessage(`"What is the capital of France?"`),
		Instructions: "Be concise.",
		Temperature:  floatPtr(0.7),
	}, server
}

func TestRoundTrip_RequestTranslation_MockBackend(t *testing.T) {
	tr, apiReq, server := buildRoundTripFixtures(t)
	defer server.Close()

	chatReq, err := tr.TranslateToCompletionRequest(apiReq, nil)
	require.NoError(t, err)

	reqBody, err := json.Marshal(chatReq)
	require.NoError(t, err)

	var wireCheck map[string]interface{}
	require.NoError(t, json.Unmarshal(reqBody, &wireCheck))

	assert.Equal(t, "gpt-4o", wireCheck["model"])
	msgs := wireCheck["messages"].([]interface{})
	require.GreaterOrEqual(t, len(msgs), 1)

	hasSystem := false
	hasUser := false
	for _, m := range msgs {
		msg := m.(map[string]interface{})
		switch msg["role"] {
		case "system":
			hasSystem = true
			assert.Equal(t, "Be concise.", msg["content"])
		case "user":
			hasUser = true
			assert.Equal(t, "What is the capital of France?", msg["content"])
		}
	}
	assert.True(t, hasSystem, "should have system message from instructions")
	assert.True(t, hasUser, "should have user message from input")
}

func TestRoundTrip_ResponseTranslation_MockBackend(t *testing.T) {
	tr, apiReq, server := buildRoundTripFixtures(t)
	defer server.Close()

	resp, err := http.Post(server.URL, "application/json", nil)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	var chatResp openai.ChatCompletion
	require.NoError(t, json.Unmarshal(respBody, &chatResp))

	apiResp := tr.TranslateToResponseAPIResponse(apiReq, &chatResp, "")

	assert.Equal(t, "response", apiResp.Object)
	assert.Equal(t, StatusCompleted, apiResp.Status)
	assert.Equal(t, "gpt-4o", apiResp.Model)
	assert.Contains(t, apiResp.OutputText, "Paris")
	require.Len(t, apiResp.Output, 1)
	assert.Equal(t, ItemTypeMessage, apiResp.Output[0].Type)
	assert.Equal(t, "assistant", apiResp.Output[0].Role)
	require.Len(t, apiResp.Output[0].Content, 1)
	assert.Equal(t, ContentTypeOutputText, apiResp.Output[0].Content[0].Type)
	assert.Contains(t, apiResp.Output[0].Content[0].Text, "Paris")

	require.NotNil(t, apiResp.Usage)
	assert.Equal(t, 15, apiResp.Usage.InputTokens)
	assert.Equal(t, 8, apiResp.Usage.OutputTokens)
	assert.Equal(t, 23, apiResp.Usage.TotalTokens)
	assert.Equal(t, "Be concise.", apiResp.Instructions)
	assert.NotEmpty(t, apiResp.ID)
	assert.Greater(t, len(apiResp.ID), 4, "response ID should be non-trivial")
}

func TestRoundTrip_ResponseAPI_ToolCalls_MockBackend(t *testing.T) {
	toolResponse := map[string]interface{}{
		"id":      "chatcmpl-mock-tools",
		"object":  "chat.completion",
		"created": 1700000000,
		"model":   "gpt-4o",
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "",
					"tool_calls": []map[string]interface{}{
						{
							"id":   "call_weather_1",
							"type": "function",
							"function": map[string]interface{}{
								"name":      "get_weather",
								"arguments": `{"city":"London","unit":"celsius"}`,
							},
						},
					},
				},
				"finish_reason": "tool_calls",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     20,
			"completion_tokens": 15,
			"total_tokens":      35,
		},
	}
	toolResponseBody, err := json.Marshal(toolResponse)
	require.NoError(t, err)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(toolResponseBody)
	}))
	defer server.Close()

	tr := NewTranslator()
	apiReq := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`"What's the weather in London?"`),
		Tools: []Tool{
			{
				Type: ToolTypeFunction,
				Function: &FunctionDef{
					Name:        "get_weather",
					Description: "Get current weather",
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"city": map[string]interface{}{"type": "string"},
							"unit": map[string]interface{}{"type": "string"},
						},
						"required": []string{"city"},
					},
				},
			},
		},
		ToolChoice: "auto",
	}

	chatReq, err := tr.TranslateToCompletionRequest(apiReq, nil)
	require.NoError(t, err)

	reqBody, err := json.Marshal(chatReq)
	require.NoError(t, err)

	var wireCheck map[string]interface{}
	require.NoError(t, json.Unmarshal(reqBody, &wireCheck))

	tools := wireCheck["tools"].([]interface{})
	require.Len(t, tools, 1)
	toolFn := tools[0].(map[string]interface{})["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", toolFn["name"])

	resp, err := http.Post(server.URL, "application/json", nil)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	var chatResp openai.ChatCompletion
	require.NoError(t, json.Unmarshal(respBody, &chatResp))

	apiResp := tr.TranslateToResponseAPIResponse(apiReq, &chatResp, "prev_resp_001")

	assert.Equal(t, "prev_resp_001", apiResp.PreviousResponseID)
	require.Len(t, apiResp.Output, 1)
	assert.Equal(t, ItemTypeFunctionCall, apiResp.Output[0].Type)
	assert.Equal(t, "get_weather", apiResp.Output[0].Name)
	assert.Equal(t, "call_weather_1", apiResp.Output[0].CallID)
	assert.Contains(t, apiResp.Output[0].Arguments, "London")
	assert.Equal(t, StatusCompleted, apiResp.Output[0].Status)
}

func TestRoundTrip_ResponseAPI_Multimodal_MockBackend(t *testing.T) {
	var capturedBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(mockChatCompletionResponse("I see a landscape.", "gpt-4o"))
	}))
	defer server.Close()

	tr := NewTranslator()
	apiReq := &ResponseAPIRequest{
		Model: "gpt-4o",
		Input: json.RawMessage(`[{
			"type": "message",
			"role": "user",
			"content": [
				{"type": "input_text", "text": "What do you see?"},
				{"type": "input_image", "image_url": "https://example.com/photo.jpg"}
			]
		}]`),
	}

	chatReq, err := tr.TranslateToCompletionRequest(apiReq, nil)
	require.NoError(t, err)

	reqBody, err := json.Marshal(chatReq)
	require.NoError(t, err)

	resp, err := http.Post(server.URL, "application/json", bytes.NewReader(reqBody))
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)

	var chatResp openai.ChatCompletion
	require.NoError(t, json.Unmarshal(respBody, &chatResp))

	apiResp := tr.TranslateToResponseAPIResponse(apiReq, &chatResp, "")
	assert.Contains(t, apiResp.OutputText, "landscape")

	var wireReq map[string]interface{}
	require.NoError(t, json.Unmarshal(capturedBody, &wireReq))
	msgs := wireReq["messages"].([]interface{})
	msg := msgs[0].(map[string]interface{})
	content := msg["content"].([]interface{})
	require.Len(t, content, 2)
	assert.Equal(t, "text", content[0].(map[string]interface{})["type"])
	imgPart := content[1].(map[string]interface{})
	assert.Equal(t, "image_url", imgPart["type"])
	imgURL := imgPart["image_url"].(map[string]interface{})
	assert.Equal(t, "https://example.com/photo.jpg", imgURL["url"])
}

func TestWireFormat_ChatCompletionRequest_SDKCompliant(t *testing.T) {
	tr := NewTranslator()
	apiReq := &ResponseAPIRequest{
		Model:           "gpt-4o",
		Input:           json.RawMessage(`"Hello"`),
		Instructions:    "Be brief.",
		Temperature:     floatPtr(0.3),
		TopP:            floatPtr(0.9),
		MaxOutputTokens: intPtr(100),
		Tools: []Tool{
			{Type: ToolTypeFunction, Function: &FunctionDef{Name: "fn1", Description: "test"}},
		},
		ToolChoice: "auto",
	}

	chatReq, err := tr.TranslateToCompletionRequest(apiReq, nil)
	require.NoError(t, err)

	body, err := json.Marshal(chatReq)
	require.NoError(t, err)

	var reparsed openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal(body, &reparsed), "must unmarshal into SDK type without error")

	assert.Equal(t, "gpt-4o", reparsed.Model)
	assert.InDelta(t, 0.3, reparsed.Temperature.Value, 0.001)
	assert.InDelta(t, 0.9, reparsed.TopP.Value, 0.001)
	assert.Equal(t, int64(100), reparsed.MaxTokens.Value)
	require.Len(t, reparsed.Tools, 1)
	assert.Equal(t, "fn1", reparsed.Tools[0].Function.Name)
}

func TestWireFormat_ChatCompletionResponse_SDKCompliant(t *testing.T) {
	raw := mockChatCompletionResponse("Test response", "gpt-4o")

	var resp openai.ChatCompletion
	require.NoError(t, json.Unmarshal(raw, &resp), "mock response must unmarshal into SDK type")

	assert.Equal(t, "chatcmpl-mock-001", resp.ID)
	assert.Equal(t, "chat.completion", string(resp.Object))
	assert.Equal(t, "gpt-4o", resp.Model)
	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "assistant", string(resp.Choices[0].Message.Role))
	assert.Equal(t, "Test response", resp.Choices[0].Message.Content)
	assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	assert.Equal(t, int64(15), resp.Usage.PromptTokens)
	assert.Equal(t, int64(8), resp.Usage.CompletionTokens)
	assert.Equal(t, int64(23), resp.Usage.TotalTokens)
}

func floatPtr(f float64) *float64 { return &f }
func intPtr(i int) *int           { return &i }
