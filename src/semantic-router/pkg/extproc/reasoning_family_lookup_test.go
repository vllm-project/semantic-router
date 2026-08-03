//go:build !windows && cgo

package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// reasoningLookupConfig: a valid config where the router model name (router-model)
// carries reasoning_family, but the upstream provider_model_id (upstream-id)
// differs — the normal case (and the shape used by the shipped config.yaml).
const reasoningLookupConfig = `
version: v0.3
listeners: []
providers:
  defaults:
    default_model: router-model
    reasoning_families:
      qwen3:
        type: chat_template_kwargs
        parameter: enable_thinking
  models:
    - name: router-model
      provider_model_id: upstream-id
      reasoning_family: qwen3
      backend_refs:
        - endpoint: 127.0.0.1:8000
          api_key: k
routing:
  modelCards:
    - name: router-model
  decisions:
    - name: d1
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: router-model
          use_reasoning: true
`

// The reasoning parameter must be injected based on the selected ROUTER model
// (which carries reasoning_family), not the rewritten upstream model id in the
// request body. Otherwise use_reasoning is silently dropped whenever
// provider_model_id/external_model_ids differ from the model name.
func TestReasoningFamilyResolvedByRoutingModelNotUpstream(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(reasoningLookupConfig))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	router := &OpenAIRouter{Config: cfg}

	// Body already has the rewritten upstream model id, as it does by the time
	// reasoning runs in the auto-routing path.
	body := []byte(`{"model":"upstream-id","messages":[{"role":"user","content":"hi"}]}`)

	out, err := router.setReasoningModeToRequestBodyForProvider(body, true, "d1", nil, "router-model")
	if err != nil {
		t.Fatalf("setReasoningModeToRequestBodyForProvider: %v", err)
	}
	if !strings.Contains(string(out), `"enable_thinking":true`) {
		t.Fatalf("expected reasoning (enable_thinking:true) injected for router-model's qwen3 family, got: %s", out)
	}
}
