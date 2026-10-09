package config

import (
	"strings"
	"testing"
)

func TestValidateRAGMaxContextLength(t *testing.T) {
	cases := []struct {
		name    string
		value   *int
		wantErr bool
	}{
		{"unset_ok", nil, false},
		{"positive_ok", intPtr(10000), false},
		{"zero_ok", intPtr(0), false},
		{"negative_rejected", intPtr(-5), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RAGPluginConfig{Enabled: true, Backend: "vectorstore", MaxContextLength: tc.value}
			err := cfg.Validate()
			if tc.wantErr != (err != nil) {
				t.Fatalf("max_context_length=%v: wantErr=%v, got err=%v", tc.value, tc.wantErr, err)
			}
			if err != nil && !strings.Contains(err.Error(), "max_context_length") {
				t.Fatalf("error should name max_context_length, got: %v", err)
			}
		})
	}
}

func TestResolveModelConfigExternalIDDeterministic(t *testing.T) {
	newConfig := func() *RouterConfig {
		return &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
			"qwen-prod": {APIFormat: "openai", ExternalModelIDs: map[string]string{"openai": "Qwen/Qwen2.5-14B-Instruct"}},
			"qwen-dev":  {APIFormat: "anthropic", ExternalModelIDs: map[string]string{"openai": "Qwen/Qwen2.5-14B-Instruct"}},
			"other":     {APIFormat: "openai"},
		}}}
	}

	// A direct model-name hit always wins over the external-id fallback.
	direct := newConfig()
	if params, ok := direct.resolveModelConfig("qwen-prod"); !ok || params.APIFormat != "openai" {
		t.Fatalf("direct lookup must return qwen-prod, got ok=%v params=%+v", ok, params)
	}

	// The external-id fallback must resolve to the same owner regardless of
	// map iteration order.
	want := "anthropic"
	for i := 0; i < 50; i++ {
		params, ok := newConfig().resolveModelConfig("Qwen/Qwen2.5-14B-Instruct")
		if !ok {
			t.Fatal("external id fallback must resolve")
		}
		if params.APIFormat != want {
			t.Fatalf("iteration %d: external id resolved to %q owner, want stable %q", i, params.APIFormat, want)
		}
	}

	if _, ok := newConfig().resolveModelConfig("unknown-model"); ok {
		t.Fatal("unknown model must not resolve")
	}
}
