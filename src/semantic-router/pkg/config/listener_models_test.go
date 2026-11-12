package config

import (
	"strings"
	"testing"
)

func TestListenerModelsAreDistinctRequestModelNames(t *testing.T) {
	for _, tt := range []struct {
		models []string
		want   string
	}{
		{nil, ""},
		{[]string{"vllm-sr/auto", "qwen/qwen3.8-27b"}, ""},
		{[]string{}, "lists no model"},
		{[]string{""}, "must be a model name"},
		{[]string{" vllm-sr/auto"}, "must be a model name"},
		{[]string{"vllm-sr/auto", "vllm-sr/auto"}, `lists "vllm-sr/auto" twice`},
	} {
		cfg := &RouterConfig{APIServer: APIServer{Listeners: []Listener{{Name: "l", Port: 8899, APIs: ListenerAPIs{"chat_completions": {Models: tt.models}}}}}}
		err := validateListenerContracts(cfg)
		if (tt.want == "") != (err == nil) || (err != nil && !strings.Contains(err.Error(), tt.want)) {
			t.Errorf("%q: error = %v, want %q", tt.models, err, tt.want)
		}
	}
}

func TestOnlyStandaloneServesListenerModels(t *testing.T) {
	cfg := &RouterConfig{APIServer: APIServer{Listeners: []Listener{
		{Name: "dashboard", Port: 8898},
		{Name: "public", Port: 8899, APIs: ListenerAPIs{"chat_completions": {Models: []string{"vllm-sr/auto"}}, "responses": {}}},
	}}}
	if err := ValidateGatewayCapabilities(cfg, GatewayStandalone); err != nil {
		t.Fatalf("standalone: %v", err)
	}
	var found []CapabilityViolation
	for _, violation := range CheckGatewayCapabilities(cfg, GatewayExtProc) {
		if violation.Capability == CapabilityListenerModels {
			found = append(found, violation)
		}
	}
	if len(found) != 1 || found[0].Path != "listeners[public].apis.chat_completions.models" ||
		!strings.Contains(found[0].Message, "listener 'public': apis.chat_completions.models is unsupported with --gateway extproc") {
		t.Fatalf("extproc violations = %+v", found)
	}
}
