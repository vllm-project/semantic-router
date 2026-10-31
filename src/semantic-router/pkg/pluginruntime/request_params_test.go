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

func TestRequestParamsBlocksForwardedFields(t *testing.T) {
	includeUsage := true
	request := &llmprotocol.Request{
		Model:         "m",
		TextVerbosity: "high",
		StreamOptions: llmprotocol.StreamOptions{IncludeUsage: &includeUsage},
		EndUserID:     "user-42",
		Sampling:      llmprotocol.Sampling{Temperature: llmprotocol.Float64(0.7)},
	}
	policy := &config.RequestParamsPluginConfig{
		BlockedParams: []string{"verbosity", "stream_options", "user", "temperature"},
	}
	result, err := ApplyRequestParams(request, policy)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Changed {
		t.Fatal("expected result.Changed to be true")
	}
	expectedBlocked := []string{"verbosity", "stream_options", "user", "temperature"}
	if len(result.Blocked) != len(expectedBlocked) {
		t.Fatalf("expected blocked %v, got %v", expectedBlocked, result.Blocked)
	}
	if request.TextVerbosity != "" {
		t.Errorf("expected TextVerbosity to be cleared, got %q", request.TextVerbosity)
	}
	if request.StreamOptions.IncludeUsage != nil {
		t.Errorf("expected StreamOptions.IncludeUsage to be cleared, got %v", *request.StreamOptions.IncludeUsage)
	}
	if request.EndUserID != "" {
		t.Errorf("expected EndUserID to be cleared, got %q", request.EndUserID)
	}
	if request.Sampling.Temperature != nil {
		t.Errorf("expected Temperature to be cleared, got %v", *request.Sampling.Temperature)
	}
}

func TestRequestParamsRejectsUnblockableParams(t *testing.T) {
	request := &llmprotocol.Request{Model: "m"}
	policy := &config.RequestParamsPluginConfig{
		BlockedParams: []string{"messages"},
	}
	_, err := ApplyRequestParams(request, policy)
	if err == nil {
		t.Fatal("expected error blocking required field messages, got nil")
	}
}
