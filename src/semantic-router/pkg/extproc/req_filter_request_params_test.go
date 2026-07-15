//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildRequestParamsMutationsNilDecision(t *testing.T) {
	r := &OpenAIRouter{}
	out, err := r.buildRequestParamsMutations(nil, []byte(`{"model":"x"}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != `{"model":"x"}` {
		t.Fatalf("expected passthrough, got %s", out)
	}
}

func TestBuildRequestParamsMutationsBlockedAndCaps(t *testing.T) {
	r := &OpenAIRouter{}
	payload, err := config.NewStructuredPayload(map[string]interface{}{
		"blocked_params":   []string{"logprobs", "custom_evil_field"},
		"max_tokens_limit": 500,
		"max_n":            1,
		"strip_unknown":    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	decision := &config.Decision{
		Name: "tier_a",
		Plugins: []config.DecisionPlugin{
			{Type: "request_params", Configuration: payload},
		},
	}
	raw := []byte(`{"model":"m","messages":[],"logprobs":true,"max_tokens":9000,"n":5,"thinking":{"type":"enabled"},"custom_evil_field":"x","foo":1}`)
	out, err := r.buildRequestParamsMutations(decision, raw, &config.ProviderProfile{
		Type:    "openai",
		BaseURL: "http://localhost:8000/v1",
	})
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]interface{}
	if err := json.Unmarshal(out, &body); err != nil {
		t.Fatal(err)
	}
	if _, ok := body["logprobs"]; ok {
		t.Fatal("expected logprobs stripped via blocked_params")
	}
	if _, ok := body["custom_evil_field"]; ok {
		t.Fatal("expected custom_evil_field stripped")
	}
	if _, ok := body["foo"]; ok {
		t.Fatal("expected unknown field stripped")
	}
	if _, ok := body["thinking"]; ok {
		t.Fatal("expected thinking stripped for non-DeepSeek provider")
	}
	if int(body["max_tokens"].(float64)) != 500 {
		t.Fatalf("max_tokens cap: got %v", body["max_tokens"])
	}
	if int(body["n"].(float64)) != 1 {
		t.Fatalf("n cap: got %v", body["n"])
	}
}

func TestBuildRequestParamsMutationsPreservesThinkingForOfficialDeepSeek(t *testing.T) {
	r := &OpenAIRouter{}
	payload, err := config.NewStructuredPayload(map[string]interface{}{
		"strip_unknown": true,
	})
	if err != nil {
		t.Fatal(err)
	}
	decision := &config.Decision{
		Name: "deepseek",
		Plugins: []config.DecisionPlugin{
			{Type: "request_params", Configuration: payload},
		},
	}
	raw := []byte(`{"model":"deepseek-chat","messages":[],"thinking":{"type":"enabled"},"foo":1}`)
	out, err := r.buildRequestParamsMutations(decision, raw, &config.ProviderProfile{
		Type:    "openai",
		BaseURL: "https://api.deepseek.com",
	})
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]interface{}
	if err := json.Unmarshal(out, &body); err != nil {
		t.Fatal(err)
	}
	if _, ok := body["foo"]; ok {
		t.Fatal("expected unknown field stripped")
	}
	if thinking, ok := body["thinking"].(map[string]interface{}); !ok || thinking["type"] != "enabled" {
		t.Fatalf("expected thinking object preserved for official DeepSeek, got %v", body["thinking"])
	}
}
