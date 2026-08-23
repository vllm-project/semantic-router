package looper

import (
	"encoding/json"

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
