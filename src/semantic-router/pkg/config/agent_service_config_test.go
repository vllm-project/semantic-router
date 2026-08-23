package config

import (
	"strings"
	"testing"
)

func TestValidateAgentServiceModeContract(t *testing.T) {
	tests := []struct {
		name     string
		mode     string
		endpoint string
		wantErr  string
	}{
		{name: "standalone omits Agent inference", mode: ControlPlaneModeStandalone},
		{
			name: "standalone rejects managed endpoint", mode: ControlPlaneModeStandalone,
			endpoint: "http://public-inference.internal/v1/chat/completions", wantErr: "managed-only",
		},
		{name: "managed requires endpoint", mode: ControlPlaneModeManaged, wantErr: "requires"},
		{
			name: "managed accepts HTTP", mode: ControlPlaneModeManaged,
			endpoint: "http://public-inference.internal/v1/chat/completions",
		},
		{
			name: "managed accepts HTTPS", mode: ControlPlaneModeManaged,
			endpoint: "https://api.example.test/v1/chat/completions",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateAgentService(test.mode, AgentServiceConfig{PublicInferenceEndpoint: test.endpoint})
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("validateAgentService() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("validateAgentService() error = %v, want fragment %q", err, test.wantErr)
			}
		})
	}
}

func TestValidateAgentServiceRejectsAmbiguousPublicEndpoints(t *testing.T) {
	for _, endpoint := range []string{
		" http://public-inference.internal/v1/chat/completions",
		"http://public-inference.internal/v1/chat/completions ",
		"grpc://public-inference.internal/v1/chat/completions",
		"http:///v1/chat/completions",
		"http://user@public-inference.internal/v1/chat/completions",
		"http://public-inference.internal/v1/chat/completions?tenant=one",
		"http://public-inference.internal/v1/chat/completions?",
		"http://public-inference.internal/v1/chat/completions#fragment",
		"http://public-inference.internal",
		"http://public-inference.internal/v1",
		"http://public-inference.internal/v1/chat/completions/",
		"http://public-inference.internal/v1%2Fchat%2Fcompletions",
	} {
		t.Run(endpoint, func(t *testing.T) {
			if err := validateAgentService(
				ControlPlaneModeManaged,
				AgentServiceConfig{PublicInferenceEndpoint: endpoint},
			); err == nil {
				t.Fatalf("validateAgentService(%q) should fail", endpoint)
			}
		})
	}
}
