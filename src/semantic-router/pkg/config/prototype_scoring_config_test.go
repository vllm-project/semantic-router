package config

import (
	"reflect"
	"testing"
)

func TestPrototypeScoringRuleOverrideDoesNotMergeFamily(t *testing.T) {
	disabled := false
	family := PrototypeScoringConfig{MaxPrototypes: 1, BestWeight: 1, TopM: 1}
	cases := []struct {
		name string
		rule *PrototypeScoringConfig
		want PrototypeScoringConfig
	}{
		{"inherit", nil, family.WithDefaults()},
		{"empty override", &PrototypeScoringConfig{}, PrototypeScoringConfig{}.WithDefaults()},
		{"retain candidates", &PrototypeScoringConfig{Enabled: &disabled}, PrototypeScoringConfig{Enabled: &disabled}.WithDefaults()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.rule.Resolve(family); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("resolved = %+v, want %+v", got, tc.want)
			}
		})
	}
	if family.Enabled != nil || disabled {
		t.Fatal("resolution mutated its source")
	}
}
