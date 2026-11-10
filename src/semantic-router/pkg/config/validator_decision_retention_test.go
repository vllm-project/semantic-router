package config

import (
	"strings"
	"testing"
)

func retentionTestBool(v bool) *bool { return &v }
func retentionTestInt(v int) *int    { return &v }

func TestValidateDecisionRetentionContracts(t *testing.T) {
	decisionWith := func(retention *DecisionRetention) *RouterConfig {
		return &RouterConfig{IntelligentRouting: IntelligentRouting{Decisions: []Decision{{Name: "r", Retention: retention}}}}
	}
	cases := []struct {
		name    string
		cfg     *RouterConfig
		wantErr string
	}{
		{
			name: "every field",
			cfg: decisionWith(&DecisionRetention{
				Drop:                  retentionTestBool(false),
				TTLTurns:              retentionTestInt(2),
				KeepCurrentModel:      retentionTestBool(true),
				PreferPrefixRetention: retentionTestBool(true),
			}),
		},
		{name: "no retention", cfg: decisionWith(nil)},
		{name: "negative ttl", cfg: decisionWith(&DecisionRetention{TTLTurns: retentionTestInt(-1)}), wantErr: "decision 'r' retention: ttl_turns must be >= 0"},
		{
			name:    "drop ttl conflict",
			cfg:     decisionWith(&DecisionRetention{Drop: retentionTestBool(true), TTLTurns: retentionTestInt(3)}),
			wantErr: "drop=true conflicts with ttl_turns",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateDecisionRetentionContracts(tc.cfg)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}
