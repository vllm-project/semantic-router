package llmprotocol

import "fmt"

// DefaultOutputTokens supplies policy only when the caller has no effective
// output bound. Model capacity metadata is never a provider default.
func DefaultOutputTokens(request *Request, value *int) bool {
	if request == nil || request.Sampling.MaxOutputTokens != nil || value == nil || *value <= 0 {
		return false
	}
	request.Sampling.MaxOutputTokens = Int64(int64(*value))
	return true
}

// DefaultAutomaticOutput preserves caller limits and defers a missing bound
// until the provider has rendered the complete model-specific input.
func DefaultAutomaticOutput(request *Request) bool {
	if request == nil || request.Sampling.MaxOutputTokens != nil || request.Sampling.AutomaticOutput {
		return false
	}
	request.Sampling.AutomaticOutput = true
	return true
}

// CapOutputTokens clamps an explicit request value; it never invents a limit.
func CapOutputTokens(request *Request, limit *int) bool {
	if request != nil && request.Sampling.AutomaticOutput && limit != nil {
		request.Sampling.AutomaticOutputCap = Int64(int64(*limit))
	}
	if limit == nil || request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens <= int64(*limit) {
		return false
	}
	request.Sampling.MaxOutputTokens = Int64(int64(*limit))
	return true
}

func CapCandidateCount(request *Request, limit *int) bool {
	if limit == nil || request.CandidateCount == nil || *request.CandidateCount <= int64(*limit) {
		return false
	}
	request.CandidateCount = Int64(int64(*limit))
	return true
}

// BlockRequestField removes a named optional field from the neutral request.
//
//nolint:cyclop,funlen // Blocking is an exhaustive mapping of the public request-parameter vocabulary.
func BlockRequestField(request *Request, field string) (bool, error) {
	switch field {
	case "":
		return false, nil
	case "model", "messages":
		return false, fmt.Errorf("required semantic field %q cannot be blocked", field)
	case "frequency_penalty":
		changed := request.Sampling.FrequencyPenalty != nil
		request.Sampling.FrequencyPenalty = nil
		return changed, nil
	case "presence_penalty":
		changed := request.Sampling.PresencePenalty != nil
		request.Sampling.PresencePenalty = nil
		return changed, nil
	case "max_tokens", "max_completion_tokens", "max_output_tokens":
		changed := request.Sampling.MaxOutputTokens != nil
		request.Sampling.MaxOutputTokens = nil
		return changed, nil
	case "n", "candidate_count":
		changed := request.CandidateCount != nil
		request.CandidateCount = nil
		return changed, nil
	case "response_format", "output_format":
		changed := request.OutputFormat.Kind != ""
		request.OutputFormat = OutputFormat{}
		return changed, nil
	case "seed":
		changed := request.Sampling.Seed != nil
		request.Sampling.Seed = nil
		return changed, nil
	case "stop":
		changed := len(request.Sampling.Stop) > 0
		request.Sampling.Stop = nil
		return changed, nil
	case "temperature":
		changed := request.Sampling.Temperature != nil
		request.Sampling.Temperature = nil
		return changed, nil
	case "top_p":
		changed := request.Sampling.TopP != nil
		request.Sampling.TopP = nil
		return changed, nil
	case "min_p":
		changed := request.Sampling.MinP != nil
		request.Sampling.MinP = nil
		return changed, nil
	case "repetition_penalty":
		changed := request.Sampling.RepetitionPenalty != nil
		request.Sampling.RepetitionPenalty = nil
		return changed, nil
	case "cache_salt":
		changed := request.CacheSalt != nil
		request.CacheSalt = nil
		return changed, nil
	case "prompt_cache_key":
		changed := request.PromptCacheKey != ""
		request.PromptCacheKey = ""
		return changed, nil
	case "top_k":
		changed := request.Sampling.TopK != nil
		request.Sampling.TopK = nil
		return changed, nil
	case "tools":
		changed := len(request.Tools) > 0
		request.Tools = nil
		return changed, nil
	case "tool_choice":
		changed := request.ToolChoice.Mode != "" || request.ToolChoice.Name != ""
		request.ToolChoice = ToolChoice{}
		return changed, nil
	case "parallel_tool_calls":
		changed := request.ParallelToolCalls != nil
		request.ParallelToolCalls = nil
		return changed, nil
	case "reasoning_effort":
		changed := request.ReasoningEffort != ""
		request.ReasoningEffort = ""
		return changed, nil
	case "reasoning_budget_tokens":
		changed := request.ReasoningBudgetTokens != nil
		request.ReasoningBudgetTokens = nil
		return changed, nil
	case "metadata":
		changed := len(request.Metadata) > 0
		request.Metadata = nil
		return changed, nil
	case "store":
		changed := request.Store != nil
		request.Store = nil
		return changed, nil
	case "stream":
		changed := request.Stream
		request.Stream = false
		return changed, nil
	default:
		// Unknown client fields never enter neutral IR; codecs already enforce
		// the configured unknown-field policy at ingress.
		return false, nil
	}
}
