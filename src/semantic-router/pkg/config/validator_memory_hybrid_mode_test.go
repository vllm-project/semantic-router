package config

import (
	"strings"
	"testing"
)

func TestValidateMemoryHybridMode(t *testing.T) {
	scopes := []struct {
		name  string
		field string
		build func(mode string) *RouterConfig
	}{
		{
			name:  "global",
			field: "global.stores.memory.hybrid_mode",
			build: func(mode string) *RouterConfig {
				cfg := &RouterConfig{}
				cfg.Memory.HybridMode = mode
				return cfg
			},
		},
	}
	modes := []struct {
		mode    string
		wantErr bool
		hint    string
	}{
		{mode: "", wantErr: false},
		{mode: "weighted", wantErr: false},
		{mode: "rrf", wantErr: false},
		{mode: "rerank", wantErr: true, hint: `use "weighted"`},
		{mode: "RRF", wantErr: true},
		{mode: " rrf", wantErr: true},
		{mode: "reciprocal_rank", wantErr: true},
	}

	for _, scope := range scopes {
		for _, tc := range modes {
			t.Run(scope.name+"/"+tc.mode, func(t *testing.T) {
				err := validateMemoryContracts(scope.build(tc.mode))
				if !tc.wantErr {
					if err != nil {
						t.Fatalf("hybrid_mode %q rejected: %v", tc.mode, err)
					}
					return
				}
				if err == nil {
					t.Fatalf("hybrid_mode %q must be rejected", tc.mode)
				}
				for _, want := range []string{scope.field, `"` + tc.mode + `"`, `"weighted", "rrf"`, tc.hint} {
					if !strings.Contains(err.Error(), want) {
						t.Fatalf("error should contain %q, got: %v", want, err)
					}
				}
			})
		}
	}
}
