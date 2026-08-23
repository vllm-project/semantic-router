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
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func semanticModelResponse(response llmprotocol.Response, model string) *llmprotocol.Response {
	cloned := cloneSemanticResponse(response)
	if cloned.Model != model {
		cloned.Model = model
		cloned.Generation++
	}
	return &cloned
}

func cloneSemanticResponse(source llmprotocol.Response) llmprotocol.Response {
	cloned := source
	cloned.Evidence.TokenLogprobs = append(
		[]llmprotocol.TokenLogprob(nil), source.Evidence.TokenLogprobs...,
	)
	for index := range cloned.Evidence.TokenLogprobs {
		cloned.Evidence.TokenLogprobs[index].Alternatives = append(
			[]llmprotocol.TokenLogprobAlternative(nil),
			source.Evidence.TokenLogprobs[index].Alternatives...,
		)
	}
	cloned.Output = cloneOutputItems(source.Output)
	cloned.Alternatives = make([][]llmprotocol.OutputItem, len(source.Alternatives))
	for index := range source.Alternatives {
		cloned.Alternatives[index] = cloneOutputItems(source.Alternatives[index])
	}
	if source.Error != nil {
		errorCopy := *source.Error
		cloned.Error = &errorCopy
	}
	return cloned
}

func cloneOutputItems(source []llmprotocol.OutputItem) []llmprotocol.OutputItem {
	cloned := append([]llmprotocol.OutputItem(nil), source...)
	for index := range cloned {
		cloned[index].Content = append([]llmprotocol.Content(nil), source[index].Content...)
		for contentIndex := range cloned[index].Content {
			content := &cloned[index].Content[contentIndex]
			if content.ToolCall != nil {
				call := *content.ToolCall
				content.ToolCall = &call
			}
			if content.ToolResult != nil {
				result := *content.ToolResult
				result.Content = append([]llmprotocol.Content(nil), content.ToolResult.Content...)
				content.ToolResult = &result
			}
		}
	}
	return cloned
}

func semanticResponseText(response *llmprotocol.Response) (content, reasoning string, hasToolCalls bool) {
	if response == nil {
		return "", "", false
	}
	for _, item := range response.Output {
		for _, block := range item.Content {
			switch block.Kind {
			case llmprotocol.ContentText, llmprotocol.ContentRefusal:
				content += block.Text
			case llmprotocol.ContentReasoning:
				reasoning += block.Text
			case llmprotocol.ContentToolCall:
				hasToolCalls = hasToolCalls || block.ToolCall != nil
			}
		}
	}
	return content, reasoning, hasToolCalls
}

func tokenUsageFromSemantic(usage llmprotocol.Usage) TokenUsage {
	if usage.State != llmprotocol.UsageAvailable || usage.InputTotal.Value == nil || usage.OutputTotal.Value == nil {
		return unknownTokenUsage()
	}
	input, output := *usage.InputTotal.Value, *usage.OutputTotal.Value
	if input < 0 || output < 0 || input > math.MaxInt64-output {
		return unknownTokenUsage()
	}
	total := input + output
	if usage.Total.Value != nil {
		total = *usage.Total.Value
	}
	return NewActualTokenUsage(input, output, total)
}

func semanticUsageFromTokenUsage(usage TokenUsage) llmprotocol.Usage {
	if !usage.isValid() {
		return llmprotocol.Usage{State: llmprotocol.UsageUnavailable}
	}
	derived := func(value int64) llmprotocol.TokenCount {
		return llmprotocol.TokenCount{Value: llmprotocol.Int64(value), Provenance: llmprotocol.UsageDerived}
	}
	unknown := llmprotocol.TokenCount{Provenance: llmprotocol.UsageUnknown}
	return llmprotocol.Usage{
		State:           llmprotocol.UsageAvailable,
		InputUncached:   unknown,
		InputCacheRead:  unknown,
		InputCacheWrite: unknown,
		OutputReasoning: unknown,
		OutputOther:     unknown,
		InputTotal:      derived(usage.PromptTokens),
		OutputTotal:     derived(usage.CompletionTokens),
		Total:           derived(usage.TotalTokens),
	}
}

func newTextSemanticResponse(idPrefix, model, content string, usage TokenUsage) llmprotocol.Response {
	id := fmt.Sprintf("%s-%d", idPrefix, time.Now().UnixNano())
	return llmprotocol.Response{
		Generation: 1,
		ID:         id,
		CreatedAt:  time.Now().UTC(),
		Model:      model,
		Output: []llmprotocol.OutputItem{{
			ID: llmprotocol.StableID(id, "0"), Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: content}},
		}},
		StopReason: llmprotocol.StopEndTurn,
		Usage:      semanticUsageFromTokenUsage(usage),
	}
}

func newTaggedToolSemanticResponse(idPrefix, model, content string, usage TokenUsage) (llmprotocol.Response, bool) {
	name, arguments, ok := parseTaggedToolCall(content)
	if !ok {
		return llmprotocol.Response{}, false
	}
	id := fmt.Sprintf("%s-%d", idPrefix, time.Now().UnixNano())
	callID := llmprotocol.StableID(id, "tool", name, arguments)
	return llmprotocol.Response{
		Generation: 1,
		ID:         id,
		CreatedAt:  time.Now().UTC(),
		Model:      model,
		Output: []llmprotocol.OutputItem{{
			ID: llmprotocol.StableID(id, "0"), Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind:     llmprotocol.ContentToolCall,
				ToolCall: &llmprotocol.ToolCall{ID: callID, Name: name, Arguments: arguments},
			}},
		}},
		StopReason: llmprotocol.StopToolCall,
		Usage:      semanticUsageFromTokenUsage(usage),
	}, true
}

func newModelSemanticResponse(
	idPrefix string,
	modelResponse *ModelResponse,
	model string,
	usage TokenUsage,
) (llmprotocol.Response, error) {
	if modelResponse == nil || modelResponse.Semantic == nil {
		return llmprotocol.Response{}, fmt.Errorf("model returned no neutral response")
	}
	response := cloneSemanticResponse(*modelResponse.Semantic)
	response.Generation++
	response.ID = fmt.Sprintf("%s-%d", idPrefix, time.Now().UnixNano())
	response.CreatedAt = time.Now().UTC()
	response.Model = model
	response.Usage = semanticUsageFromTokenUsage(usage)
	response.Evidence = llmprotocol.ResponseEvidence{}
	if response.StopReason == "" || response.StopReason == llmprotocol.StopUnknown {
		response.StopReason = llmprotocol.StopEndTurn
	}
	for itemIndex := range response.Output {
		if strings.TrimSpace(response.Output[itemIndex].ID) == "" {
			response.Output[itemIndex].ID = llmprotocol.StableID(response.ID, fmt.Sprint(itemIndex))
		}
	}
	return response, nil
}

func newLooperResponse(
	semantic llmprotocol.Response,
	streaming bool,
	includeUsage bool,
	model string,
	modelsUsed []string,
	iterations int,
	algorithm string,
	usage TokenUsage,
	intermediate interface{},
) *Response {
	return &Response{
		Semantic:              &semantic,
		Streaming:             streaming,
		IncludeUsage:          includeUsage,
		Model:                 model,
		ModelsUsed:            append([]string(nil), modelsUsed...),
		Iterations:            iterations,
		AlgorithmType:         algorithm,
		IntermediateResponses: intermediate,
		Usage:                 usage,
	}
}
