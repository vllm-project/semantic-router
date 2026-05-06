package config

import (
	"strings"
	"testing"
)

func TestValidateRandomContracts(t *testing.T) {
	tests := []struct {
		name    string
		rules   []RandomRule
		wantErr string
	}{
		{
			name:  "valid",
			rules: []RandomRule{{Name: "random_digit"}},
		},
		{
			name:    "empty name",
			rules:   []RandomRule{{}},
			wantErr: "name cannot be empty",
		},
		{
			name:    "duplicate name",
			rules:   []RandomRule{{Name: "random_digit"}, {Name: "random_digit"}},
			wantErr: "duplicate rule name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateRandomContracts(&RouterConfig{
				IntelligentRouting: IntelligentRouting{
					Signals: Signals{RandomRules: tt.rules},
				},
			})
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("validateRandomContracts() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("validateRandomContracts() error = %v, want containing %q", err, tt.wantErr)
			}
		})
	}
}
