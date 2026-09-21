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
