package pluginruntime

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestRequestParamsBlockedOutputIsDefaultedBeforeCap(t *testing.T) {
	defaultTokens, maxTokens, maxN := 100, 50, 1
	request := &llmprotocol.Request{CandidateCount: llmprotocol.Int64(3), Sampling: llmprotocol.Sampling{MaxOutputTokens: llmprotocol.Int64(1000)}}
	result, err := ApplyRequestParams(request, &config.RequestParamsPluginConfig{BlockedParams: []string{"max_tokens"}, DefaultMaxTokens: config.FixedOutputTokenDefault(defaultTokens), MaxTokensLimit: &maxTokens, MaxN: &maxN})
	if err != nil {
		t.Fatal(err)
	}
	if *request.Sampling.MaxOutputTokens != 50 || *request.CandidateCount != 1 || !result.DefaultedOutputTokens || !result.CappedOutputTokens || !result.CappedCandidateCount {
		t.Fatalf("wrong policy effects: %+v, request %+v", result, request)
	}
}

// A route whose backends run speculative decoding can remove the two sampling
// controls such a vLLM backend rejects.
func TestRequestParamsBlocksMinPAndLogitBias(t *testing.T) {
	minP := 0.05
	request := &llmprotocol.Request{Sampling: llmprotocol.Sampling{MinP: &minP, LogitBias: map[string]float64{"42": -100}}}
	result, err := ApplyRequestParams(request, &config.RequestParamsPluginConfig{BlockedParams: []string{"min_p", "logit_bias"}})
	if err != nil {
		t.Fatal(err)
	}
	if request.Sampling.MinP != nil || request.Sampling.LogitBias != nil || !result.Changed ||
		len(result.Blocked) != 2 || result.Blocked[0] != "min_p" || result.Blocked[1] != "logit_bias" {
		t.Fatalf("request params did not block min_p and logit_bias: sampling=%+v result=%+v", request.Sampling, result)
	}
}

func TestRequestParamsBlocksPromptCacheKey(t *testing.T) {
	request := &llmprotocol.Request{PromptCacheKey: "codex-session"}
	result, err := ApplyRequestParams(request, &config.RequestParamsPluginConfig{BlockedParams: []string{"prompt_cache_key"}})
	if err != nil {
		t.Fatal(err)
	}
	if request.PromptCacheKey != "" || !result.Changed || len(result.Blocked) != 1 || result.Blocked[0] != "prompt_cache_key" {
		t.Fatalf("request params did not block prompt_cache_key: request=%+v result=%+v", request, result)
	}
}
