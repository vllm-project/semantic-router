package config

import "testing"

func TestValidateFusionGroundingConfigEarlyExitBounds(t *testing.T) {
	tests := []struct {
		name      string
		threshold float64
		wantErr   bool
	}{
		{"zero ok", 0.0, false},
		{"mid ok", 0.85, false},
		{"one ok", 1.0, false},
		{"negative rejected", -0.1, true},
		{"above one rejected", 1.5, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &FusionGroundingConfig{
				Enabled:                 true,
				EarlyExitEnabled:        true,
				EarlyExitMinConsistency: tt.threshold,
			}
			err := ValidateFusionGroundingConfig(cfg)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ValidateFusionGroundingConfig(threshold=%v) err=%v, wantErr=%v", tt.threshold, err, tt.wantErr)
			}
		})
	}
}

func TestValidateFusionEscalationConfig(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *FusionEscalationConfig
		wantErr bool
	}{
		{"nil ok", nil, false},
		{"disabled ignores empty rules", &FusionEscalationConfig{Enabled: false}, false},
		{"enabled with rule ok", &FusionEscalationConfig{Enabled: true, HardComplexityRules: []string{"x:hard"}}, false},
		{"enabled without rules rejected", &FusionEscalationConfig{Enabled: true}, true},
		{"enabled with only-blank rules rejected", &FusionEscalationConfig{Enabled: true, HardComplexityRules: []string{"  "}}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateFusionEscalationConfig(tt.cfg); (err != nil) != tt.wantErr {
				t.Fatalf("ValidateFusionEscalationConfig err=%v, wantErr=%v", err, tt.wantErr)
			}
		})
	}
}
