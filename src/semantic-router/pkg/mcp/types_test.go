package mcp

import (
	"encoding/json"
	"testing"

	mcpsdk "github.com/mark3labs/mcp-go/mcp"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConvertToolToOpenAI_BasicTool(t *testing.T) {
	mcpTool := mcpsdk.Tool{
		Name:        "get_weather",
		Description: "Get weather for a city",
		InputSchema: mcpsdk.ToolInputSchema{
			Type: "object",
			Properties: map[string]interface{}{
				"city": map[string]interface{}{
					"type":        "string",
					"description": "City name",
				},
			},
			Required: []string{"city"},
		},
	}

	result := ConvertToolToOpenAI(mcpTool)

	assert.Equal(t, "get_weather", result.Function.Name)
	assert.Equal(t, "Get weather for a city", result.Function.Description.Value)

	body, err := json.Marshal(result)
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &parsed))

	assert.Equal(t, "function", parsed["type"])
	fn := parsed["function"].(map[string]interface{})
	assert.Equal(t, "get_weather", fn["name"])
	params := fn["parameters"].(map[string]interface{})
	assert.Equal(t, "object", params["type"])
	props := params["properties"].(map[string]interface{})
	city := props["city"].(map[string]interface{})
	assert.Equal(t, "string", city["type"])
}

func TestConvertToolToOpenAI_NoParams(t *testing.T) {
	mcpTool := mcpsdk.Tool{
		Name:        "ping",
		Description: "Ping the server",
	}

	result := ConvertToolToOpenAI(mcpTool)

	assert.Equal(t, "ping", result.Function.Name)
	assert.Equal(t, "Ping the server", result.Function.Description.Value)

	body, err := json.Marshal(result)
	require.NoError(t, err)

	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &parsed))

	fn := parsed["function"].(map[string]interface{})
	_, hasParams := fn["parameters"]
	assert.False(t, hasParams, "tool without schema should not have parameters key")
}

func TestConvertOpenAIToMCPCall_Simple(t *testing.T) {
	toolCall := openai.ChatCompletionMessageToolCall{
		ID:   "call_abc123",
		Type: "function",
		Function: openai.ChatCompletionMessageToolCallFunction{
			Name:      "get_weather",
			Arguments: `{"city":"London"}`,
		},
	}

	result, err := ConvertOpenAIToMCPCall(toolCall)
	require.NoError(t, err)

	assert.Equal(t, "get_weather", result.Params.Name)
	args, ok := result.Params.Arguments.(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "London", args["city"])
}

func TestConvertOpenAIToMCPCall_EmptyArgs(t *testing.T) {
	toolCall := openai.ChatCompletionMessageToolCall{
		ID:   "call_empty",
		Type: "function",
		Function: openai.ChatCompletionMessageToolCallFunction{
			Name:      "ping",
			Arguments: "",
		},
	}

	result, err := ConvertOpenAIToMCPCall(toolCall)
	require.NoError(t, err)

	assert.Equal(t, "ping", result.Params.Name)
	assert.Empty(t, result.Params.Arguments)
}

func TestConvertOpenAIToMCPCall_InvalidJSON(t *testing.T) {
	toolCall := openai.ChatCompletionMessageToolCall{
		ID:   "call_bad",
		Type: "function",
		Function: openai.ChatCompletionMessageToolCallFunction{
			Name:      "bad_tool",
			Arguments: `{not valid json`,
		},
	}

	_, err := ConvertOpenAIToMCPCall(toolCall)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse arguments")
}

func TestConvertToolToOpenAI_RoundTrip_JSON(t *testing.T) {
	mcpTool := mcpsdk.Tool{
		Name:        "search",
		Description: "Search the web",
		InputSchema: mcpsdk.ToolInputSchema{
			Type: "object",
			Properties: map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search query",
				},
				"limit": map[string]interface{}{
					"type":        "number",
					"description": "Max results",
				},
			},
			Required: []string{"query"},
		},
	}

	openaiTool := ConvertToolToOpenAI(mcpTool)

	body, err := json.Marshal(openaiTool)
	require.NoError(t, err)

	var reparsed openai.ChatCompletionToolParam
	require.NoError(t, json.Unmarshal(body, &reparsed))

	assert.Equal(t, "search", reparsed.Function.Name)
}

func TestConvertMCPResultToOpenAI_TextContent(t *testing.T) {
	result := &mcpsdk.CallToolResult{
		Content: []mcpsdk.Content{
			mcpsdk.TextContent{
				Type: "text",
				Text: "The weather is sunny",
			},
		},
	}

	converted := ConvertMCPResultToOpenAI(result)
	assert.Equal(t, "The weather is sunny", converted["content"])
}

func TestConvertMCPResultToOpenAI_NilResult(t *testing.T) {
	converted := ConvertMCPResultToOpenAI(nil)
	assert.Equal(t, "No result", converted["content"])
}
