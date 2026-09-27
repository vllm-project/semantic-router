package looper

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"
)

// prepareModelCallBody applies the actual outbound mutations and admission
// checks. Planner selection can use this same path without making a model call.
func prepareModelCallBody(req *openai.ChatCompletionNewParams, target ModelTarget, options CallOptions) ([]byte, error) {
	// Clone and modify the request with the target model
	modifiedReq := cloneRequest(req)
	modifiedReq.Model = target.Name
	if err := capLooperStageOutput(options.candidateRequest, modifiedReq); err != nil {
		return nil, err
	}

	// Configure logprobs based on config
	if options.Logprobs != nil && options.Logprobs.Enabled {
		modifiedReq.Logprobs = openai.Bool(true)
		topLogprobs := options.Logprobs.TopLogprobs
		if topLogprobs < 1 {
			topLogprobs = 1 // Need at least 1 for margin calculation
		}
		if topLogprobs > 5 {
			topLogprobs = 5 // API limit
		}
		modifiedReq.TopLogprobs = openai.Int(int64(topLogprobs))
	}

	// Marshal request to JSON first
	body, err := json.Marshal(modifiedReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Add stream parameter via JSON manipulation (SDK doesn't expose Stream field)
	streaming := options.Mode == ResponseSSE
	body, err = setStreamParam(body, streaming)
	if err != nil {
		return nil, fmt.Errorf("failed to set stream param: %w", err)
	}
	if err := validateLooperStageCandidate(options.candidateRequest, body, target.Name, options.Logprobs); err != nil {
		return nil, err
	}

	return body, nil
}
