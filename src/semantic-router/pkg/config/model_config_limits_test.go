package config

import (
	"testing"
	"time"

	"gopkg.in/yaml.v2"
)

func TestExternalModelTimeoutSecondsKeepsFractions(t *testing.T) {
	cases := []struct {
		name string
		yaml string
		want time.Duration
	}{
		{name: "fractional", yaml: "llm_timeout_seconds: 0.8\n", want: 800 * time.Millisecond},
		{name: "integer", yaml: "llm_timeout_seconds: 5\n", want: 5 * time.Second},
		{name: "omitted", yaml: "llm_model_name: classifier\n", want: 30 * time.Second},
		{name: "zero", yaml: "llm_timeout_seconds: 0\n", want: 30 * time.Second},
		{name: "negative", yaml: "llm_timeout_seconds: -1\n", want: 30 * time.Second},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var cfg ExternalModelConfig
			if err := yaml.Unmarshal([]byte(tc.yaml), &cfg); err != nil {
				t.Fatalf("yaml.Unmarshal() error = %v", err)
			}
			if got := cfg.GetTimeout(); got != tc.want {
				t.Fatalf("GetTimeout() = %v, want %v (parsed llm_timeout_seconds = %v)", got, tc.want, cfg.TimeoutSeconds)
			}
		})
	}
}
