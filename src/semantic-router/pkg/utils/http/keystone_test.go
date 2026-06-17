package http

import (
	"encoding/json"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"github.com/openai/openai-go"
)

// headerValueByKey returns the value of the first SetHeader matching key, or "".
func headerValueByKey(opts []*core.HeaderValueOption, key string) string {
	for _, h := range opts {
		if h.Header.Key == key {
			return string(h.Header.RawValue)
		}
	}
	return ""
}

// TestKeystoneHeaderOptions pins the wire contract of the shared keystone
// builder: exactly two options, schema-version "2", and the given response path.
func TestKeystoneHeaderOptions(t *testing.T) {
	opts := KeystoneHeaderOptions("cache")
	if len(opts) != 2 {
		t.Fatalf("expected 2 keystone options, got %d", len(opts))
	}
	if v := headerValueByKey(opts, "x-vsr-schema-version"); v != "2" {
		t.Errorf("schema-version: expected \"2\", got %q", v)
	}
	if v := headerValueByKey(opts, "x-vsr-response-path"); v != "cache" {
		t.Errorf("response-path: expected \"cache\", got %q", v)
	}
}

// TestCacheHitResponseEmitsKeystone verifies the cache immediate response
// carries schema-version "2" and response-path "cache".
func TestCacheHitResponseEmitsKeystone(t *testing.T) {
	completion := openai.ChatCompletion{
		ID:     "chatcmpl-keystone-cache",
		Object: "chat.completion",
		Model:  "test-model",
		Choices: []openai.ChatCompletionChoice{
			{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "hi"}, FinishReason: "stop"},
		},
	}
	body, err := json.Marshal(completion)
	if err != nil {
		t.Fatalf("Failed to marshal cached response: %v", err)
	}

	set := CreateCacheHitResponse(body, false, "math", "math_decision", nil).GetImmediateResponse().Headers.SetHeaders
	if v := headerValueByKey(set, "x-vsr-schema-version"); v != "2" {
		t.Errorf("cache schema-version: expected \"2\", got %q", v)
	}
	if v := headerValueByKey(set, "x-vsr-response-path"); v != "cache" {
		t.Errorf("cache response-path: expected \"cache\", got %q", v)
	}
}

// TestFastResponseEmitsKeystone verifies the fast_response immediate response
// carries schema-version "2" and response-path "fast_response".
func TestFastResponseEmitsKeystone(t *testing.T) {
	set := CreateFastResponse("blocked message", false, "guard_decision").GetImmediateResponse().Headers.SetHeaders
	if v := headerValueByKey(set, "x-vsr-schema-version"); v != "2" {
		t.Errorf("fast schema-version: expected \"2\", got %q", v)
	}
	if v := headerValueByKey(set, "x-vsr-response-path"); v != "fast_response" {
		t.Errorf("fast response-path: expected \"fast_response\", got %q", v)
	}
}
