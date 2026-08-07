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

import "github.com/openai/openai-go"

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
