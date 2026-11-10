package looper

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"
)

func stripFusionToolUse(req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
	if req == nil {
		return nil
	}
	stripped := cloneRequest(req)
	stripped.Tools = nil
	stripped.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{}
	stripped.Functions = nil
	stripped.FunctionCall = openai.ChatCompletionNewParamsFunctionCallUnion{}
	return stripped
}

func appendFusionStageMessage(req *openai.ChatCompletionNewParams, content string) *openai.ChatCompletionNewParams {
	if req == nil {
		return nil
	}
	data, err := json.Marshal(req)
	if err != nil {
		return req
	}
	var reqMap map[string]interface{}
	err = json.Unmarshal(data, &reqMap)
	if err != nil {
		return req
	}
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		return req
	}
	reqMap["messages"] = append(messages, map[string]string{
		"role":    "user",
		"content": content,
	})
	data, err = json.Marshal(reqMap)
	if err != nil {
		return req
	}
	var appended openai.ChatCompletionNewParams
	err = json.Unmarshal(data, &appended)
	if err != nil {
		return req
	}
	return &appended
}

// buildStreamingToolCallSSE streams an answer that calls tools: the text the
// model wrote beside the calls, then the calls.
func buildStreamingToolCallSSE(id string, created int64, resp *ModelResponse) ([]byte, error) {
	toolCalls, err := fusionToolCallDeltasFromRaw(resp.Raw)
	if err != nil {
		return nil, err
	}
	return buildSimulatedChatCompletionSSE(id, created, resp.Model, splitIntoChunks(resp.Content, 50), toolCalls, backendFinishReason(resp)), nil
}

func fusionToolCallDeltasFromRaw(raw []byte) ([]map[string]interface{}, error) {
	var completion struct {
		Choices []struct {
			Message struct {
				ToolCalls []map[string]interface{} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &completion); err != nil {
		return nil, fmt.Errorf("failed to parse fusion streaming tool-call response: %w", err)
	}
	if len(completion.Choices) == 0 || len(completion.Choices[0].Message.ToolCalls) == 0 {
		return nil, fmt.Errorf("fusion streaming tool-call response did not contain tool_calls")
	}

	toolCalls := make([]map[string]interface{}, 0, len(completion.Choices[0].Message.ToolCalls))
	for i, toolCall := range completion.Choices[0].Message.ToolCalls {
		delta := make(map[string]interface{}, len(toolCall)+1)
		for key, value := range toolCall {
			delta[key] = value
		}
		delta["index"] = i
		toolCalls = append(toolCalls, delta)
	}
	return toolCalls, nil
}
