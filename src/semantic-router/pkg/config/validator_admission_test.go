package config

import (
	"strings"
	"testing"
)

func TestValidateModelAdmissionContracts(t *testing.T) {
	tests := []struct {
		name      string
		key       string
		admission AdmissionConfig
		wantErr   string
	}{
		{"valid", DefaultDecisionDeployment, AdmissionConfig{MaxConcurrency: 4, MaxQueue: 64, QueueTimeoutMs: 250, OnOverflow: "shed"}, ""},
		{"defaults", EmbeddingDefaultDeployment, AdmissionConfig{MaxConcurrency: 1}, ""},
		{"unknown deployment", "unknown_model", AdmissionConfig{MaxConcurrency: 4}, "unknown deployment"},
		{"zero concurrency", DefaultDecisionDeployment, AdmissionConfig{}, "max_concurrency"},
		{"negative queue", DefaultDecisionDeployment, AdmissionConfig{MaxConcurrency: 1, MaxQueue: -1}, "max_queue"},
		{"negative timeout", DefaultDecisionDeployment, AdmissionConfig{MaxConcurrency: 1, QueueTimeoutMs: -1}, "queue_timeout_ms"},
		{"invalid overflow", DefaultDecisionDeployment, AdmissionConfig{MaxConcurrency: 1, OnOverflow: "drop"}, "on_overflow"},
		{"wait without queue", DefaultDecisionDeployment, AdmissionConfig{MaxConcurrency: 1, OnOverflow: "wait"}, "max_queue >= 1"},
		{"wait with queue", DefaultDecisionDeployment, AdmissionConfig{MaxConcurrency: 1, MaxQueue: 8, OnOverflow: "wait"}, ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.ModelDeployments = DefaultModelDeployments()
			cfg.ModelAdmission = map[string]AdmissionConfig{test.key: test.admission}
			err := validateModelAdmissionContracts(cfg)
			if test.wantErr == "" && err != nil {
				t.Fatal(err)
			}
			if test.wantErr != "" && (err == nil || !strings.Contains(err.Error(), test.wantErr)) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}
