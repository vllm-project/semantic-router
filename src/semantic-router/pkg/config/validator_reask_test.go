package config

import (
	"strings"
	"testing"
)

func TestValidateReaskContractsAcceptsDefaults(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				ReaskRules: []ReaskRule{{
					Name: "likely_dissatisfied",
				}},
			},
		},
	}

	if err := validateReaskContracts(cfg); err != nil {
		t.Fatalf("validateReaskContracts() error = %v", err)
	}
}

func TestValidateReaskContractsRejectsInvalidThreshold(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				ReaskRules: []ReaskRule{{
					Name:      "likely_dissatisfied",
					Threshold: 1.1,
				}},
			},
		},
	}

	err := validateReaskContracts(cfg)
	if err == nil {
		t.Fatal("expected error for invalid threshold")
	}
	if !strings.Contains(err.Error(), "threshold must be between 0 and 1") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateReaskContractsRejectsInvalidLookbackTurns(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				ReaskRules: []ReaskRule{{
					Name:          "persistently_dissatisfied",
					LookbackTurns: -1,
				}},
			},
		},
	}

	err := validateReaskContracts(cfg)
	if err == nil {
		t.Fatal("expected error for invalid lookback_turns")
	}
	if !strings.Contains(err.Error(), "lookback_turns must be >= 1") {
		t.Fatalf("unexpected error: %v", err)
	}
}
