package looper

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelpricing"
)

func TestUsageAccountingRetainsCacheBucketsAndRejectsUnknown(t *testing.T) {
	body := []byte(`{"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_tokens_details":{"cached_tokens":40},"cache_creation_input_tokens":20}}`)
	usage := parseResponseUsage(body)
	if !usage.Complete() || usage.CachedInputTokens != 40 || usage.CacheWriteTokens != 20 {
		t.Fatalf("cache usage lost: %+v", usage)
	}
	aggregate := usage.Add(&ModelResponse{Usage: usage})
	if aggregate.CachedInputTokens != 80 || aggregate.CacheWriteTokens != 40 {
		t.Fatal("cache totals lost")
	}
	wire, _ := json.Marshal(usage.Map())
	var roundtrip TokenUsage
	if err := json.Unmarshal(wire, &roundtrip); err != nil || roundtrip != usage {
		t.Fatalf("roundtrip: %+v %v", roundtrip, err)
	}
	for _, input := range []string{`{}`, `{"usage":null}`, `{"usage":{"prompt_tokens":100}}`, `{"usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20,"cached_input_tokens":11}}`, `{"usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20,"cached_input_tokens":0,"prompt_tokens_details":{"cached_tokens":3}}}`} {
		if u := parseResponseUsage([]byte(input)); u.Complete() {
			t.Fatalf("invalid usage accepted: %s", input)
		}
	}
	if actualAttemptCost(modelpricing.Rates{}, TokenUsage{Unreported: true}) != nil {
		t.Fatal("unknown cost priced")
	}
}

func TestUsageAccountingStreamingFinalInvalidDoesNotReuseEarlierUsage(t *testing.T) {
	body := []byte("data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1,\"total_tokens\":11}}\n\ndata: {\"usage\":{\"prompt_tokens\":\"bad\"}}\n\ndata: [DONE]\n")
	if parseStreamingUsage(body).Complete() {
		t.Fatal("invalid final usage retained previous counts")
	}
	if parseStreamingUsage([]byte("data: [DONE]\n")).Complete() {
		t.Fatal("absent streaming usage marked known")
	}
}
