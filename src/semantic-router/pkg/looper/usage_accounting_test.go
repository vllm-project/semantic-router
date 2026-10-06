package looper

import (
	"encoding/json"
	"fmt"
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

func TestUsageAccountingCacheWriteAliases(t *testing.T) {
	for _, tc := range []struct {
		name, fields string
		complete     bool
	}{
		{"provider created cache", `"prompt_tokens_details":{"created_cache_tokens":20}`, true},
		{"provider cache creation", `"prompt_tokens_details":{"cache_creation_tokens":20}`, true},
		{"matching aliases", `"cache_write_tokens":20,"cache_creation_input_tokens":20,"prompt_tokens_details":{"cache_write_tokens":20,"cache_creation_tokens":20,"created_cache_tokens":20}`, true},
		{"conflicting creation aliases", `"prompt_tokens_details":{"cache_creation_tokens":19,"created_cache_tokens":20}`, false},
		{"conflicting detail write", `"prompt_tokens_details":{"cache_write_tokens":19,"created_cache_tokens":20}`, false},
		{"conflicting top level write", `"cache_write_tokens":0,"prompt_tokens_details":{"created_cache_tokens":20}`, false},
		{"conflicting top level creation", `"cache_creation_input_tokens":19,"prompt_tokens_details":{"cache_creation_tokens":20}`, false},
		{"negative write", `"prompt_tokens_details":{"created_cache_tokens":-1}`, false},
		{"write exceeds remaining prompt", `"prompt_tokens_details":{"cached_tokens":90,"created_cache_tokens":20}`, false},
		{"noninteger write", `"prompt_tokens_details":{"created_cache_tokens":20.5}`, false},
		{"string write", `"prompt_tokens_details":{"cache_creation_tokens":"20"}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := fmt.Appendf(nil, `{"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,%s}}`, tc.fields)
			for name, usage := range map[string]TokenUsage{
				"json": parseResponseUsage(body),
				"sse":  parseStreamingUsage(fmt.Appendf(nil, "data: %s\n\ndata: [DONE]\n\n", body)),
			} {
				if usage.Complete() != tc.complete || tc.complete && usage.CacheWriteTokens != 20 {
					t.Errorf("%s usage=%+v complete=%t, want complete=%t", name, usage, usage.Complete(), tc.complete)
				}
				if !tc.complete && actualAttemptCost(modelpricing.Rates{Currency: "USD", PromptPer1M: 1}, usage) != nil {
					t.Errorf("%s priced incomplete or conflicting cache usage", name)
				}
			}
		})
	}
}
