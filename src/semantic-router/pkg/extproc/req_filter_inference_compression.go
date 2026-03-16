package extproc

// req_filter_inference_compression.go — Use Case 2: pre-inference compression.
//
// Compresses individual messages in the request body BEFORE forwarding to
// vLLM, using a domain-specific Pipeline selected by the routing decision.
//
// Flow:
//   performDecisionEvaluation (UC1)
//     → EvaluateAllSignals → DecisionResult (domain is now known)
//       → handleModelRouting → modifyRequestBodyForAutoRouting
//           → compressMessagesInBody  ← this file
//               → vLLM
//
// The domain is extracted from decisionName (the Name of the matched
// Decision). The PipelineSelector maps decision names to Pipelines loaded
// from YAML at startup. Each Pipeline's MaxTokens controls the per-message
// compression budget: messages already shorter than MaxTokens are passed
// through unchanged.
//
// System messages are never compressed — they define model behavior and are
// typically already concise. Only user, assistant, and tool messages are
// eligible.
//
// Contrast with Use Case 1 (req_filter_classification.go):
//
//   UC1: domain unknown, single global Config, compresses for signal latency.
//   UC2: domain known, PipelineSelector, compresses for inference latency.

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/promptcompression"
)

// compressMessagesInBody applies per-message NLP compression to the request
// body JSON, targeting messages that individually exceed the pipeline's
// MaxTokens budget.
//
// Algorithm:
//  1. Unmarshal the body to access the messages array.
//  2. For each non-system message: if its token count exceeds MaxTokens,
//     compress with CompressWithPipeline using the domain-specific pipeline.
//  3. If any message was compressed, re-marshal and return the smaller body.
//     Otherwise return the original body bytes unchanged.
//
// The per-message budget (pipeline.MaxTokens) is distinct from the UC1
// classification budget. Each domain YAML typically sets a larger value,
// e.g. max_tokens: 2048 for coding (preserve code context) vs.
// max_tokens: 1024 for general topics.
//
// System messages are never modified. Array-of-parts content (vision/multimodal)
// is left untouched — only plain string content is eligible for compression.
func compressMessagesInBody(body []byte, pipeline promptcompression.Pipeline) ([]byte, error) {
	if pipeline.MaxTokens <= 0 {
		return body, nil
	}

	var request map[string]interface{}
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, fmt.Errorf("inference compression: unmarshal body: %w", err)
	}

	messages, ok := request["messages"].([]interface{})
	if !ok || len(messages) == 0 {
		return body, nil
	}

	compressed, changed := compressMessageSlice(messages, pipeline)
	if !changed {
		return body, nil
	}

	request["messages"] = compressed
	out, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("inference compression: marshal body: %w", err)
	}
	return out, nil
}

// compressMessageSlice iterates over the decoded messages array and compresses
// any text-only, non-system message whose content exceeds the budget.
// Returns the (possibly modified) slice and whether any change was made.
func compressMessageSlice(
	messages []interface{},
	pipeline promptcompression.Pipeline,
) ([]interface{}, bool) {
	out := make([]interface{}, len(messages))
	changed := false

	for i, raw := range messages {
		msg, ok := raw.(map[string]interface{})
		if !ok {
			out[i] = raw
			continue
		}

		role, _ := msg["role"].(string)
		content, isString := msg["content"].(string)

		// System messages and multimodal (non-string) content are never touched.
		if role == "system" || role == "developer" || !isString || content == "" {
			out[i] = raw
			continue
		}

		tokens := promptcompression.CountTokensApprox(content)
		if tokens <= pipeline.MaxTokens {
			out[i] = raw
			continue
		}

		result := promptcompression.CompressWithPipeline(content, pipeline)
		if result.Ratio >= 1.0 {
			// No compression achieved (e.g. single sentence input).
			out[i] = raw
			continue
		}

		// Build a new map to avoid mutating the original.
		newMsg := make(map[string]interface{}, len(msg))
		for k, v := range msg {
			newMsg[k] = v
		}
		newMsg["content"] = result.Compressed
		out[i] = newMsg
		changed = true

		logging.Infof(
			"[InferenceCompression] role=%s: %d→%d tokens (ratio=%.2f)",
			role, result.OriginalTokens, result.CompressedTokens, result.Ratio,
		)
	}

	return out, changed
}
