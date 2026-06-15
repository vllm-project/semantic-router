//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildRequestParamsMutationsNilDecision(t *testing.T) {
	r := &OpenAIRouter{}
	out, err := r.buildRequestParamsMutations(nil, []byte(`{"model":"x"}`))
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
	raw := []byte(`{"model":"m","messages":[],"logprobs":true,"max_tokens":9000,"n":5,"custom_evil_field":"x","foo":1}`)
	out, err := r.buildRequestParamsMutations(decision, raw)
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
	if int(body["max_tokens"].(float64)) != 500 {
		t.Fatalf("max_tokens cap: got %v", body["max_tokens"])
	}
	if int(body["n"].(float64)) != 1 {
		t.Fatalf("n cap: got %v", body["n"])
	}
}
