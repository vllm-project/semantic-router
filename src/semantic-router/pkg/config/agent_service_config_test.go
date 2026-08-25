package config

import (
	"strings"
	"testing"
)

func TestValidateAgentServiceAccessContract(t *testing.T) {
	tests := []struct {
		name     string
		access   bool
		endpoint string
		wantErr  string
	}{
		{name: "access disabled omits Agent inference"},
		{
			name:     "access disabled rejects Agent endpoint",
			endpoint: "http://public-inference.internal/v1/chat/completions", wantErr: "access.enabled",
		},
		{name: "access enabled requires endpoint", access: true, wantErr: "requires"},
		{
			name: "access enabled accepts HTTP", access: true,
			endpoint: "http://public-inference.internal/v1/chat/completions",
		},
		{
			name: "access enabled accepts HTTPS", access: true,
			endpoint: "https://api.example.test/v1/chat/completions",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateAgentService(test.access, AgentServiceConfig{PublicInferenceEndpoint: test.endpoint})
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
				true,
				AgentServiceConfig{PublicInferenceEndpoint: endpoint},
			); err == nil {
				t.Fatalf("validateAgentService(%q) should fail", endpoint)
			}
		})
	}
}
