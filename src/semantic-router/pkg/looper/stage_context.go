/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

// StageContextWindowError reports a Looper-generated request that no longer
// fits a candidate's configured context window. Request content is never
// included in the error.
type StageContextWindowError struct {
	Model           string
	EstimatedTokens int
	ContextWindow   int
}

func (e *StageContextWindowError) Error() string {
	return fmt.Sprintf(
		"looper stage for model %q requires an estimated %d context tokens, exceeding its configured window %d",
		e.Model,
		e.EstimatedTokens,
		e.ContextWindow,
	)
}

func (l *BaseLooper) callModelWithContextGate(
	ctx context.Context,
	baseReq *Request,
	stageReq *openai.ChatCompletionNewParams,
	modelName string,
	streaming bool,
	iteration int,
	logprobsConfig *LogprobsConfig,
	accessKey string,
) (*ModelResponse, error) {
	if err := validateLooperStageContext(baseReq, stageReq, modelName); err != nil {
		return nil, err
	}
	return l.client.CallModel(
		ctx,
		stageReq,
		modelName,
		streaming,
		iteration,
		logprobsConfig,
		accessKey,
	)
}

func validateLooperStageContext(
	baseReq *Request,
	stageReq *openai.ChatCompletionNewParams,
	modelName string,
) error {
	if baseReq == nil || baseReq.BaseContextTokens <= 0 || stageReq == nil {
		return nil
	}
	window := looperModelContextWindow(baseReq, modelName)
	if window <= 0 {
		return nil
	}
	addedTokens, err := looperStageAddedMessageTokens(baseReq.OriginalRequest, stageReq)
	if err != nil {
		return fmt.Errorf("estimate looper stage context for model %q: %w", modelName, err)
	}
	baseOutputReserve := looperOutputTokenReserve(baseReq.OriginalRequest)
	stageOutputReserve := looperOutputTokenReserve(stageReq)
	if stageOutputReserve > baseOutputReserve {
		addedTokens = saturatingAddContextTokens(
			addedTokens,
			stageOutputReserve-baseOutputReserve,
		)
	}
	estimated := saturatingAddContextTokens(baseReq.BaseContextTokens, addedTokens)
	if estimated > window {
		return &StageContextWindowError{
			Model:           modelName,
			EstimatedTokens: estimated,
			ContextWindow:   window,
		}
	}
	return nil
}

func looperOutputTokenReserve(req *openai.ChatCompletionNewParams) int {
	if req == nil {
		return 0
	}
	if req.MaxCompletionTokens.Value > 0 {
		return saturatingInt64ToInt(req.MaxCompletionTokens.Value)
	}
	if req.MaxTokens.Value > 0 {
		return saturatingInt64ToInt(req.MaxTokens.Value)
	}
	return 0
}

func saturatingInt64ToInt(value int64) int {
	maxInt := int(^uint(0) >> 1)
	if value > int64(maxInt) {
		return maxInt
	}
	return int(value)
}

func saturatingAddContextTokens(left, right int) int {
	if right <= 0 {
		return left
	}
	maxInt := int(^uint(0) >> 1)
	if left > maxInt-right {
		return maxInt
	}
	return left + right
}

func looperModelContextWindow(req *Request, modelName string) int {
	if req == nil || req.ModelParams == nil {
		return 0
	}
	if params, ok := req.ModelParams[modelName]; ok {
		return params.ContextWindowSize
	}
	for _, ref := range req.ModelRefs {
		if ref.Model == modelName || ref.LoRAName == modelName {
			return req.ModelParams[ref.Model].ContextWindowSize
		}
	}
	for _, params := range req.ModelParams {
		for _, externalID := range params.ExternalModelIDs {
			if externalID == modelName {
				return params.ContextWindowSize
			}
		}
	}
	return 0
}

func looperStageAddedMessageTokens(
	original *openai.ChatCompletionNewParams,
	stage *openai.ChatCompletionNewParams,
) (int, error) {
	if original == nil {
		return serializedMessagesTokenEstimate(stage.Messages)
	}
	prefix, err := messagesStartWith(stage.Messages, original.Messages)
	if err != nil {
		return 0, err
	}
	if prefix {
		return serializedMessagesTokenEstimate(stage.Messages[len(original.Messages):])
	}
	// Replacement stages such as self-verification do not preserve the
	// original message prefix. Counting the full replacement on top of the
	// conservative base estimate intentionally errs on the safe side.
	return serializedMessagesTokenEstimate(stage.Messages)
}

func messagesStartWith(
	messages []openai.ChatCompletionMessageParamUnion,
	prefix []openai.ChatCompletionMessageParamUnion,
) (bool, error) {
	if len(messages) < len(prefix) {
		return false, nil
	}
	for i := range prefix {
		left, err := json.Marshal(messages[i])
		if err != nil {
			return false, err
		}
		right, err := json.Marshal(prefix[i])
		if err != nil {
			return false, err
		}
		if string(left) != string(right) {
			return false, nil
		}
	}
	return true, nil
}

func serializedMessagesTokenEstimate(messages []openai.ChatCompletionMessageParamUnion) (int, error) {
	if len(messages) == 0 {
		return 0, nil
	}
	encoded, err := json.Marshal(messages)
	if err != nil {
		return 0, err
	}
	return contextcompression.EstimateTokens(string(encoded)), nil
}
