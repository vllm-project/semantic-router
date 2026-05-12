package config

import (
	"strings"
	"testing"
)

func retentionTestBool(v bool) *bool { return &v }
func retentionTestInt(v int) *int    { return &v }

func TestValidateDecisionEmitContracts(t *testing.T) {
	cases := []struct {
		name    string
		cfg     *RouterConfig
		wantErr string
	}{
		{
			name: "valid retention directive",
			cfg: &RouterConfig{IntelligentRouting: IntelligentRouting{Decisions: []Decision{{
				Name: "r",
				Emits: []EmitDirective{{
					Kind: emitDirectiveKindRetention,
					Retention: &RetentionDirective{
						Drop:                  retentionTestBool(false),
						TTLTurns:              retentionTestInt(2),
						KeepCurrentModel:      retentionTestBool(true),
						PreferPrefixRetention: retentionTestBool(true),
					},
				}},
			}}}},
		},
		{
			name: "unsupported kind",
			cfg: &RouterConfig{IntelligentRouting: IntelligentRouting{Decisions: []Decision{{
				Name:  "r",
				Emits: []EmitDirective{{Kind: "bogus"}},
			}}}},
			wantErr: "unsupported EMIT kind",
		},
		{
			name: "duplicate retention",
			cfg: &RouterConfig{IntelligentRouting: IntelligentRouting{Decisions: []Decision{{
				Name: "r",
				Emits: []EmitDirective{
					{Kind: emitDirectiveKindRetention, Retention: &RetentionDirective{Drop: retentionTestBool(false)}},
					{Kind: emitDirectiveKindRetention, Retention: &RetentionDirective{TTLTurns: retentionTestInt(1)}},
				},
			}}}},
			wantErr: "duplicate EMIT kind",
		},
		{
			name: "missing retention payload",
			cfg: &RouterConfig{IntelligentRouting: IntelligentRouting{Decisions: []Decision{{
				Name:  "r",
				Emits: []EmitDirective{{Kind: emitDirectiveKindRetention}},
			}}}},
			wantErr: "retention payload is required",
		},
		{
			name: "negative ttl",
			cfg: &RouterConfig{IntelligentRouting: IntelligentRouting{Decisions: []Decision{{
				Name: "r",
				Emits: []EmitDirective{{
					Kind:      emitDirectiveKindRetention,
					Retention: &RetentionDirective{TTLTurns: retentionTestInt(-1)},
				}},
			}}}},
			wantErr: "ttl_turns must be >= 0",
		},
		{
			name: "drop ttl conflict",
			cfg: &RouterConfig{IntelligentRouting: IntelligentRouting{Decisions: []Decision{{
				Name: "r",
				Emits: []EmitDirective{{
					Kind: emitDirectiveKindRetention,
					Retention: &RetentionDirective{
						Drop:     retentionTestBool(true),
						TTLTurns: retentionTestInt(3),
					},
				}},
			}}}},
			wantErr: "drop=true conflicts with ttl_turns",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateDecisionEmitContracts(tc.cfg)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("validateDecisionEmitContracts() unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("validateDecisionEmitContracts() error = %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}
