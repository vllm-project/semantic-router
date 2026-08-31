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
		{"valid", "prompt_guard", AdmissionConfig{MaxConcurrency: 4, MaxQueue: 64, QueueTimeoutMs: 250, OnOverflow: "shed"}, ""},
		{"defaults", "pii_classifier", AdmissionConfig{MaxConcurrency: 1}, ""},
		{"unknown deployment", "unknown_model", AdmissionConfig{MaxConcurrency: 4}, "unknown deployment"},
		{"zero concurrency", "prompt_guard", AdmissionConfig{}, "max_concurrency"},
		{"negative queue", "prompt_guard", AdmissionConfig{MaxConcurrency: 1, MaxQueue: -1}, "max_queue"},
		{"negative timeout", "prompt_guard", AdmissionConfig{MaxConcurrency: 1, QueueTimeoutMs: -1}, "queue_timeout_ms"},
		{"invalid overflow", "prompt_guard", AdmissionConfig{MaxConcurrency: 1, OnOverflow: "drop"}, "on_overflow"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := &RouterConfig{}
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
