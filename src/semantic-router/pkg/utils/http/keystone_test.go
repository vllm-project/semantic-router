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
// builder: exactly two options, schema-version "3", and the given response path.
func TestKeystoneHeaderOptions(t *testing.T) {
	opts := KeystoneHeaderOptions("cache")
	if len(opts) != 2 {
		t.Fatalf("expected 2 keystone options, got %d", len(opts))
	}
	if v := headerValueByKey(opts, "x-vsr-schema-version"); v != "3" {
		t.Errorf("schema-version: expected \"3\", got %q", v)
	}
	if v := headerValueByKey(opts, "x-vsr-response-path"); v != "cache" {
		t.Errorf("response-path: expected \"cache\", got %q", v)
	}
}

// The immediate responses carry their content-type and keystone headers
// only; the Router adds the rest of the public set from the request.
func TestImmediateResponsesCarryOnlyTheirKeystone(t *testing.T) {
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
	for _, tt := range []struct {
		name string
		set  []*core.HeaderValueOption
		path string
	}{
		{"cache hit", CreateCacheHitResponse(body, false).GetImmediateResponse().Headers.SetHeaders, "cache"},
		{"fast response", CreateFastResponse("blocked message", false).GetImmediateResponse().Headers.SetHeaders, "fast_response"},
	} {
		if len(tt.set) != 3 || headerValueByKey(tt.set, "content-type") != "application/json" {
			t.Errorf("%s: headers = %v, want content-type and the keystone", tt.name, tt.set)
		}
		if v := headerValueByKey(tt.set, "x-vsr-schema-version"); v != "3" {
			t.Errorf("%s schema-version: expected \"3\", got %q", tt.name, v)
		}
		if v := headerValueByKey(tt.set, "x-vsr-response-path"); v != tt.path {
			t.Errorf("%s response-path: expected %q, got %q", tt.name, tt.path, v)
		}
	}
}
