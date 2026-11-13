package config

import (
	"reflect"
	"strings"
	"testing"
)

func unsupportedCapabilitiesDocument(names string) []byte {
	return []byte(`
version: v0.5
providers:
  models:
    - name: draft-model
      provider_model_id: draft-model
      api_format: chat_completions
      unsupported_capabilities: ` + names + `
      backend_refs:
        - name: draft-backend
          endpoint: 127.0.0.1:18001
          protocol: http
          provider: vllm
    - name: plain-model
      provider_model_id: plain-model
      api_format: chat_completions
      backend_refs:
        - name: plain-backend
          endpoint: 127.0.0.1:18002
          protocol: http
          provider: vllm
recipes:
  - name: default
    decisions:
      - name: chat
        priority: 10
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: draft-model
          - model: plain-model
        plugins:
          - type: request_params
            configuration:
              blocked_params: [logit_bias]
`)
}

// A model whose backends reject min_p and logit_bias, as a vLLM backend that
// runs speculative decoding does, declares them; the declaration reaches the
// runtime model and survives export. Any other name fails at load and names
// the ones a model may declare.
func TestUnsupportedCapabilitiesDeclareOnlySamplingControls(t *testing.T) {
	cfg, err := ParseYAMLBytes(unsupportedCapabilitiesDocument("[sampling_min_p, sampling_logit_bias]"))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"sampling_min_p", "sampling_logit_bias"}
	if got := cfg.ModelConfig["draft-model"].UnsupportedCapabilities; !reflect.DeepEqual(got, want) {
		t.Fatalf("draft-model unsupported capabilities = %v, want %v", got, want)
	}
	if got := cfg.ModelConfig["plain-model"].UnsupportedCapabilities; len(got) != 0 {
		t.Fatalf("plain-model unsupported capabilities = %v, want none", got)
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	for _, model := range exported.Providers.Models {
		if model.Name == "draft-model" && !reflect.DeepEqual(model.UnsupportedCapabilities, want) {
			t.Fatalf("exported draft-model unsupported capabilities = %v, want %v", model.UnsupportedCapabilities, want)
		}
	}

	for _, names := range []string{"[image_input]", "[sampling_top_k]", "[min_p]", "[Sampling_Min_P]"} {
		_, err := ParseYAMLBytes(unsupportedCapabilitiesDocument(names))
		if err == nil || !strings.Contains(err.Error(), "providers.models[draft-model].unsupported_capabilities") ||
			!strings.Contains(err.Error(), "use one of: sampling_min_p, sampling_logit_bias") {
			t.Fatalf("unsupported_capabilities %s: error = %v", names, err)
		}
	}
}
