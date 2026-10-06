package config

import (
	"strings"
	"testing"
)

func TestProgressGateEffectiveConfigAndValidation(t *testing.T) {
	ptr := func(v int) *int { return &v }
	on := true
	for _, tc := range []struct {
		name string
		gate *ProgressGateTuning
		want string
	}{
		{"defaults", nil, ""},
		{"observe", &ProgressGateTuning{Enabled: &on}, ""},
		{"empty_window", &ProgressGateTuning{WindowSize: ptr(0)}, "window_size"},
		{"empty_ttl", &ProgressGateTuning{WindowTTLSeconds: ptr(0)}, "window_ttl_seconds"},
		{"oversized", &ProgressGateTuning{WindowSize: ptr(257)}, "window_size"},
		{"short_window", &ProgressGateTuning{WindowSize: ptr(2)}, "thresholds"},
		{"default_recovery", &ProgressGateTuning{MinConsecutiveRegressions: ptr(1)}, "min_consecutive_regressions"},
		{"uncalibrated_enforce", &ProgressGateTuning{Enabled: &on, Mode: "enforce"}, "calibration_id"},
		{"unversioned", &ProgressGateTuning{CalibrationID: "profile"}, "calibration_id"},
		{"versioned", &ProgressGateTuning{Enabled: &on, Mode: " enforce ", CalibrationID: "example/profile@1.0.0"}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateProgressGateTuning("progress_gate", tc.gate)
			if tc.want == "" && err != nil || tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
	cfg := (*ProgressGateTuning)(nil).EffectiveConfig()
	if cfg.Enabled || cfg.Mode != "observe" || cfg.WindowSize != 8 || cfg.WindowTTLSeconds != 900 {
		t.Fatalf("defaults changed: %+v", cfg)
	}
}
