package main

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

func TestPanelCachePreservesAccountingAndUnknownUsage(t *testing.T) {
	for _, unreported := range []bool{false, true} {
		usage := looper.TokenUsage{
			PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120,
			CachedInputTokens: 30, CacheWriteTokens: 10, Unreported: unreported,
		}
		encoded, err := json.Marshal([]cachedResponse{{Model: "test-model", Usage: recordUsage(usage)}})
		if err != nil {
			t.Fatal(err)
		}
		var cached []cachedResponse
		if err := json.Unmarshal(encoded, &cached); err != nil {
			t.Fatal(err)
		}
		responses := toModelResponses(cached)
		if len(responses) != 1 || responses[0].Usage != usage {
			t.Fatalf("accounting changed across cache: %+v", responses)
		}
		if responses[0].Usage.Complete() == unreported {
			t.Fatalf("unknown usage became reported: %+v", responses[0].Usage)
		}
	}
}
