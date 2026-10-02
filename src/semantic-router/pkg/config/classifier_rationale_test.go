package config

import (
	"strings"
	"testing"
)

func TestClassifierDisableRationaleRequiresLLM(t *testing.T) {
	for _, ruleType := range []string{ClassifierSignalTypeLocal, ClassifierSignalTypeSequenceClassifier} {
		for _, provider := range []string{"candle", "http"} {
			t.Run(ruleType+"/"+provider, func(t *testing.T) {
				cfg := genericBindingConfig(provider, ruleType)
				cfg.ClassifierRules[0].DisableRationale = true
				if _, err := CompileModelBindings(cfg); err == nil || !strings.Contains(err.Error(), "disable_rationale") {
					t.Fatalf("bound classifier error = %v", err)
				}
				cfg.ModelBindings = nil
				rule := &cfg.ClassifierRules[0]
				rule.Model = "new-endpoint"
				if ruleType == ClassifierSignalTypeLocal {
					rule.Model = ""
				}
				if err := validateClassifierSignalContracts(cfg); err == nil || !strings.Contains(err.Error(), "disable_rationale") {
					t.Fatalf("unbound classifier error = %v", err)
				}
			})
		}
	}
}
