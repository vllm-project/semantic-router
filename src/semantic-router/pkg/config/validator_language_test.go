package config

import (
	"strings"
	"testing"
)

func TestValidateLanguageContractsAcceptsSupportedISOCodes(t *testing.T) {
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{Signals: Signals{LanguageRules: []LanguageRule{
		{Name: "en"},
		{Name: "zh", Threshold: 0.5},
		{Name: "ja", Threshold: 1},
	}}}}
	if err := validateLanguageContracts(cfg); err != nil {
		t.Fatalf("valid language rules rejected: %v", err)
	}
}

func TestValidateLanguageContractsRejectsNamesThatCannotMatch(t *testing.T) {
	for _, name := range []string{"unified_frontier_zh", "ZH", "xx", ""} {
		t.Run(name, func(t *testing.T) {
			cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{Signals: Signals{LanguageRules: []LanguageRule{{Name: name}}}}}
			err := validateLanguageContracts(cfg)
			if err == nil || !strings.Contains(err.Error(), "ISO 639-1") {
				t.Fatalf("language name %q: expected ISO code error, got %v", name, err)
			}
		})
	}
}

func TestValidateLanguageContractsRejectsInvalidThreshold(t *testing.T) {
	cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{Signals: Signals{LanguageRules: []LanguageRule{{Name: "en", Threshold: 1.1}}}}}
	if err := validateLanguageContracts(cfg); err == nil || !strings.Contains(err.Error(), "between 0 and 1") {
		t.Fatalf("expected threshold error, got %v", err)
	}
}
