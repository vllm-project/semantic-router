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
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ToolStageGate decides which tools one Looper attempt may send. stage is the
// attempt's trusted-facts role; an empty stage names no role. It returns the
// request to send and must not mutate req.
type ToolStageGate func(stage llmprotocol.TrustedStage, req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams

// WithoutToolUse returns a copy of req with no tools, tool choice, or legacy
// function calls.
func WithoutToolUse(req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
	return stripFusionToolUse(req)
}

// NarrowToolUse returns a copy of req keeping only tools that pass the
// decision's allow and block lists. When it removes any tool, the tool choice
// is cleared so a forced choice cannot point at a removed tool.
func NarrowToolUse(req *openai.ChatCompletionNewParams, allowTools, blockTools []string) *openai.ChatCompletionNewParams {
	if req == nil {
		return nil
	}
	allowSet := make(map[string]bool, len(allowTools))
	for _, name := range allowTools {
		allowSet[name] = true
	}
	blockSet := make(map[string]bool, len(blockTools))
	for _, name := range blockTools {
		blockSet[name] = true
	}
	kept := make([]openai.ChatCompletionToolParam, 0, len(req.Tools))
	for _, tool := range req.Tools {
		name := tool.Function.Name
		if blockSet[name] || (len(allowSet) > 0 && !allowSet[name]) {
			continue
		}
		kept = append(kept, tool)
	}
	if len(kept) == len(req.Tools) {
		return req
	}
	narrowed := cloneRequest(req)
	narrowed.Tools = kept
	narrowed.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{}
	return narrowed
}

// toolFreeLooperRequest creates a private request for algorithms whose output
// cannot represent or resume multiple independent tool trajectories. Workflow
// owns stateful tools; Fusion and Confidence preserve only their final selected
// model's tool call. ReMoM, Ratings, RL-driven, and the fallback Base looper
// must generate self-contained textual candidates.
func toolFreeLooperRequest(req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
	return stripFusionToolUse(cloneRequest(req))
}

// normalizeCompletionToolFinishReason repairs OpenAI-compatible backends that
// return structured tool_calls with finish_reason="stop". Clients use
// finish_reason to decide whether they must execute tools, so the wrapper must
// publish the canonical value even when the upstream omits it.
func normalizeCompletionToolFinishReason(completion map[string]interface{}) {
	choices, ok := completion["choices"].([]interface{})
	if !ok {
		return
	}
	for _, rawChoice := range choices {
		choice, ok := rawChoice.(map[string]interface{})
		if !ok {
			continue
		}
		message, ok := choice["message"].(map[string]interface{})
		if !ok {
			continue
		}
		toolCalls, ok := message["tool_calls"].([]interface{})
		if ok && len(toolCalls) > 0 {
			choice["finish_reason"] = "tool_calls"
		}
	}
}
