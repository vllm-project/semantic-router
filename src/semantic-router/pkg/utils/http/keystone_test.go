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

// hasHeaderKey reports whether any SetHeader carries the given key.
func hasHeaderKey(opts []*core.HeaderValueOption, key string) bool {
	for _, h := range opts {
		if h.Header.Key == key {
			return true
		}
	}
	return false
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

// TestCacheHitResponseDemotesDetailHeaders pins the omit-when-empty contract the
// v0.4 demote relies on (#2205): the cache immediate response always carries the
// cache-hit marker and the selected-decision fact, while the category,
// cache-similarity and matched-keyword detail headers are emitted only when
// populated. The req_filter_cache caller demotes them by passing empty values
// unless the request opted into x-vsr-debug.
func TestCacheHitResponseDemotesDetailHeaders(t *testing.T) {
	body := []byte(`{"id":"x","object":"chat.completion","choices":[]}`)
	demoted := []string{"x-vsr-selected-category", "x-vsr-cache-similarity", "x-vsr-matched-keywords"}

	// Lean surface (empty detail, as the non-debug caller passes): demoted.
	lean := CreateCacheHitResponse(body, false, "", "math_decision", nil).
		GetImmediateResponse().Headers.SetHeaders
	if v := headerValueByKey(lean, "x-vsr-cache-hit"); v != "true" {
		t.Errorf("cache-hit: expected \"true\", got %q", v)
	}
	if v := headerValueByKey(lean, "x-vsr-selected-decision"); v != "math_decision" {
		t.Errorf("selected-decision: expected \"math_decision\", got %q", v)
	}
	for _, key := range demoted {
		if hasHeaderKey(lean, key) {
			t.Errorf("lean surface must omit demoted header %q", key)
		}
	}

	// Populated detail (as the debug caller passes): the headers re-appear.
	full := CreateCacheHitResponse(body, false, "math", "math_decision", []string{"prove", "theorem"}, 0.93).
		GetImmediateResponse().Headers.SetHeaders
	if v := headerValueByKey(full, "x-vsr-selected-category"); v != "math" {
		t.Errorf("category: expected \"math\", got %q", v)
	}
	if v := headerValueByKey(full, "x-vsr-cache-similarity"); v != "0.9300" {
		t.Errorf("cache-similarity: expected \"0.9300\", got %q", v)
	}
	if v := headerValueByKey(full, "x-vsr-matched-keywords"); v != "prove,theorem" {
		t.Errorf("matched-keywords: expected \"prove,theorem\", got %q", v)
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
