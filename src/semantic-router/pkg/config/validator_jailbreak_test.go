package config

import "testing"

func jailbreakCfg(rules ...JailbreakRule) *RouterConfig {
	cfg := &RouterConfig{}
	cfg.JailbreakRules = rules
	return cfg
}

// `method: hybrid` ships in config/config.yaml with jailbreak_patterns, but the
// runtime only reads patterns on the contrastive path and sends every other
// method to the model. Loading such a rule silently discards the patterns.
func TestValidateJailbreakContractsRejectsUnknownMethod(t *testing.T) {
	err := validateJailbreakContracts(jailbreakCfg(JailbreakRule{
		Name: "prompt_injection", Method: "hybrid", Threshold: 0.8,
	}))
	if err == nil {
		t.Fatal("unknown method must be rejected, not silently downgraded to the model path")
	}
}

func TestValidateJailbreakContractsAcceptsKnownMethods(t *testing.T) {
	for _, method := range []string{"", "classifier", "model", "contrastive"} {
		rule := JailbreakRule{Name: "r", Method: method, Threshold: 0.5}
		if err := validateJailbreakContracts(jailbreakCfg(rule)); err != nil {
			t.Errorf("method %q must be accepted: %v", method, err)
		}
	}
}

func TestValidateJailbreakContractsRejectsPatternsOnModelMethod(t *testing.T) {
	err := validateJailbreakContracts(jailbreakCfg(JailbreakRule{
		Name: "r", Method: "classifier", Threshold: 0.5,
		JailbreakPatterns: []string{"ignore previous instructions"},
	}))
	if err == nil {
		t.Fatal("patterns on a model-backed rule must be rejected; they are never evaluated")
	}
}

// riskScore >= threshold, so threshold 0 (or an omitted threshold, which is the
// same value in Go) marks every request as a jailbreak match. Verified at
// runtime: a rule with threshold 0 matched "What is the capital of France?"
// with a risk score of 0.0000011.
func TestValidateJailbreakContractsRejectsNonPositiveThreshold(t *testing.T) {
	for _, th := range []float32{0, -0.1} {
		err := validateJailbreakContracts(jailbreakCfg(JailbreakRule{
			Name: "r", Method: "classifier", Threshold: th,
		}))
		if err == nil {
			t.Errorf("threshold %v must be rejected: it matches every request", th)
		}
	}
}

func TestValidateJailbreakContractsRejectsThresholdAboveOne(t *testing.T) {
	if err := validateJailbreakContracts(jailbreakCfg(JailbreakRule{
		Name: "r", Threshold: 1.5,
	})); err == nil {
		t.Fatal("threshold > 1 must be rejected: it can never match")
	}
}

func TestValidateJailbreakContractsRejectsDuplicateAndUnnamedRules(t *testing.T) {
	if err := validateJailbreakContracts(jailbreakCfg(
		JailbreakRule{Name: "a", Threshold: 0.5},
		JailbreakRule{Name: "a", Threshold: 0.5},
	)); err == nil {
		t.Fatal("duplicate rule names must be rejected")
	}
	if err := validateJailbreakContracts(jailbreakCfg(JailbreakRule{Threshold: 0.5})); err == nil {
		t.Fatal("unnamed rule must be rejected")
	}
}

func TestValidateJailbreakContractsAcceptsContrastiveWithPatterns(t *testing.T) {
	if err := validateJailbreakContracts(jailbreakCfg(JailbreakRule{
		Name: "r", Method: "contrastive", Threshold: 0.5,
		JailbreakPatterns: []string{"ignore previous instructions"},
		BenignPatterns:    []string{"explain the policy"},
	})); err != nil {
		t.Fatalf("contrastive with patterns must be accepted: %v", err)
	}
}

// The runtime dispatches on the raw string, so a differently spelled method
// would load and then silently take the model path.
func TestValidateJailbreakContractsRejectsNonCanonicalSpelling(t *testing.T) {
	for _, method := range []string{"CONTRASTIVE", "Contrastive", " contrastive ", "Classifier"} {
		err := validateJailbreakContracts(jailbreakCfg(JailbreakRule{
			Name: "r", Method: method, Threshold: 0.5,
		}))
		if err == nil {
			t.Errorf("method %q must be rejected; the runtime matches %q exactly", method, "contrastive")
		}
	}
}
