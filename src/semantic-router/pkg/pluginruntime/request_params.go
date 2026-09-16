// Package pluginruntime contains side-effect-free policy operations shared by
// provider dispatch and management previews. Persistence and network calls
// remain owned by the runtime services.
package pluginruntime

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type RequestParamsResult struct {
	Changed               bool     `json:"changed"`
	Blocked               []string `json:"blocked,omitempty"`
	DefaultedOutputTokens bool     `json:"defaulted_output_tokens"`
	CappedOutputTokens    bool     `json:"capped_output_tokens"`
	CappedCandidateCount  bool     `json:"capped_candidate_count"`
}

// ApplyRequestParams implements the dispatch ordering: remove explicitly
// blocked parameters, then apply output defaults and caps.
func ApplyRequestParams(request *llmprotocol.Request, policy *config.RequestParamsPluginConfig) (RequestParamsResult, error) {
	result := RequestParamsResult{}
	if request == nil || policy == nil {
		return result, nil
	}
	for _, field := range policy.BlockedParams {
		blocked, err := llmprotocol.BlockRequestField(request, strings.TrimSpace(field))
		if err != nil {
			return result, err
		}
		if blocked {
			result.Blocked = append(result.Blocked, field)
		}
	}
	result.DefaultedOutputTokens = llmprotocol.DefaultOutputTokens(request, policy.DefaultMaxTokens)
	result.CappedOutputTokens = llmprotocol.CapOutputTokens(request, policy.MaxTokensLimit)
	result.CappedCandidateCount = llmprotocol.CapCandidateCount(request, policy.MaxN)
	result.Changed = len(result.Blocked) > 0 || result.DefaultedOutputTokens || result.CappedOutputTokens || result.CappedCandidateCount
	return result, nil
}
