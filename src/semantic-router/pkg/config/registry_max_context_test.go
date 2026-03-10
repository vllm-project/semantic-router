package config

import (
	"testing"
)

// TestMaxContextLength_SafeValues verifies that MaxContextLength reflects
// the actual training context length (512 tokens) for BERT LoRA models,
// while BaseModelMaxContext indicates the base model's capability (32K)
func TestMaxContextLength_SafeValues(t *testing.T) {
	// Test Domain Classifier
	domainModel := GetModelByPath("models/mom-domain-classifier")
	if domainModel == nil {
		t.Fatal("Domain classifier model not found")
	}
	// MaxContextLength should be 512 (trained context length)
	if domainModel.MaxContextLength != 512 {
		t.Errorf("Domain Classifier: Expected MaxContextLength=512 (trained length), got %d", domainModel.MaxContextLength)
	}
	// BaseModelMaxContext should be 32768 (base model capability)
	if domainModel.BaseModelMaxContext != 32768 {
		t.Errorf("Domain Classifier: Expected BaseModelMaxContext=32768 (base model supports 32K), got %d", domainModel.BaseModelMaxContext)
	}

	// Test PII Detector
	piiModel := GetModelByPath("models/mom-pii-classifier")
	if piiModel == nil {
		t.Fatal("PII detector model not found")
	}
	if piiModel.MaxContextLength != 512 {
		t.Errorf("PII Detector: Expected MaxContextLength=512 (trained length), got %d", piiModel.MaxContextLength)
	}
	if piiModel.BaseModelMaxContext != 32768 {
		t.Errorf("PII Detector: Expected BaseModelMaxContext=32768 (base model supports 32K), got %d", piiModel.BaseModelMaxContext)
	}

	// Test Jailbreak Classifier
	jailbreakModel := GetModelByPath("models/mom-jailbreak-classifier")
	if jailbreakModel == nil {
		t.Fatal("Jailbreak classifier model not found")
	}
	if jailbreakModel.MaxContextLength != 512 {
		t.Errorf("Jailbreak Classifier: Expected MaxContextLength=512 (trained length), got %d", jailbreakModel.MaxContextLength)
	}
	if jailbreakModel.BaseModelMaxContext != 32768 {
		t.Errorf("Jailbreak Classifier: Expected BaseModelMaxContext=32768 (base model supports 32K), got %d", jailbreakModel.BaseModelMaxContext)
	}
}

// TestMaxContextLength_ByAlias verifies that MaxContextLength and BaseModelMaxContext
// are correct when accessing models by their aliases
func TestMaxContextLength_ByAlias(t *testing.T) {
	// Test Domain Classifier by alias
	domainByAlias := GetModelByPath("domain-classifier")
	if domainByAlias == nil {
		t.Fatal("Domain classifier not found by alias")
	}
	if domainByAlias.MaxContextLength != 512 {
		t.Errorf("Domain Classifier (by alias): Expected MaxContextLength=512, got %d", domainByAlias.MaxContextLength)
	}
	if domainByAlias.BaseModelMaxContext != 32768 {
		t.Errorf("Domain Classifier (by alias): Expected BaseModelMaxContext=32768, got %d", domainByAlias.BaseModelMaxContext)
	}

	// Test PII Detector by alias
	piiByAlias := GetModelByPath("pii-detector")
	if piiByAlias == nil {
		t.Fatal("PII detector not found by alias")
	}
	if piiByAlias.MaxContextLength != 512 {
		t.Errorf("PII Detector (by alias): Expected MaxContextLength=512, got %d", piiByAlias.MaxContextLength)
	}
	if piiByAlias.BaseModelMaxContext != 32768 {
		t.Errorf("PII Detector (by alias): Expected BaseModelMaxContext=32768, got %d", piiByAlias.BaseModelMaxContext)
	}

	// Test Jailbreak Classifier by alias
	jailbreakByAlias := GetModelByPath("jailbreak-detector")
	if jailbreakByAlias == nil {
		t.Fatal("Jailbreak classifier not found by alias")
	}
	if jailbreakByAlias.MaxContextLength != 512 {
		t.Errorf("Jailbreak Classifier (by alias): Expected MaxContextLength=512, got %d", jailbreakByAlias.MaxContextLength)
	}
	if jailbreakByAlias.BaseModelMaxContext != 32768 {
		t.Errorf("Jailbreak Classifier (by alias): Expected BaseModelMaxContext=32768, got %d", jailbreakByAlias.BaseModelMaxContext)
	}
}

// TestMaxContextLength_BaseModelContext verifies that models with BaseModelMaxContext
// have appropriate descriptions explaining the distinction between trained and base model context
func TestMaxContextLength_BaseModelContext(t *testing.T) {
	domainModel := GetModelByPath("models/mom-domain-classifier")
	if domainModel == nil {
		t.Fatal("Domain classifier model not found")
	}
	if domainModel.BaseModelMaxContext > 0 {
		// Check that description explains the context length distinction
		desc := domainModel.Description
		hasContextInfo := contains(desc, "512") || contains(desc, "trained") || contains(desc, "32K") || contains(desc, "32768")
		if !hasContextInfo {
			t.Errorf("Domain Classifier description should mention context length information, got: %s", desc)
		}
		// MaxContextLength should be less than or equal to BaseModelMaxContext
		if domainModel.MaxContextLength > domainModel.BaseModelMaxContext {
			t.Errorf("Domain Classifier: MaxContextLength (%d) should be <= BaseModelMaxContext (%d)",
				domainModel.MaxContextLength, domainModel.BaseModelMaxContext)
		}
	}

	piiModel := GetModelByPath("models/mom-pii-classifier")
	if piiModel == nil {
		t.Fatal("PII detector model not found")
	}
	if piiModel.BaseModelMaxContext > 0 {
		desc := piiModel.Description
		hasContextInfo := contains(desc, "512") || contains(desc, "trained") || contains(desc, "32K") || contains(desc, "32768")
		if !hasContextInfo {
			t.Errorf("PII Detector description should mention context length information, got: %s", desc)
		}
		if piiModel.MaxContextLength > piiModel.BaseModelMaxContext {
			t.Errorf("PII Detector: MaxContextLength (%d) should be <= BaseModelMaxContext (%d)",
				piiModel.MaxContextLength, piiModel.BaseModelMaxContext)
		}
	}

	jailbreakModel := GetModelByPath("models/mom-jailbreak-classifier")
	if jailbreakModel == nil {
		t.Fatal("Jailbreak classifier model not found")
	}
	if jailbreakModel.BaseModelMaxContext > 0 {
		desc := jailbreakModel.Description
		hasContextInfo := contains(desc, "512") || contains(desc, "trained") || contains(desc, "32K") || contains(desc, "32768")
		if !hasContextInfo {
			t.Errorf("Jailbreak Classifier description should mention context length information, got: %s", desc)
		}
		if jailbreakModel.MaxContextLength > jailbreakModel.BaseModelMaxContext {
			t.Errorf("Jailbreak Classifier: MaxContextLength (%d) should be <= BaseModelMaxContext (%d)",
				jailbreakModel.MaxContextLength, jailbreakModel.BaseModelMaxContext)
		}
	}
}

// Helper function to check if string contains substring (case-insensitive)
func contains(s, substr string) bool {
	return len(s) >= len(substr) &&
		(s == substr ||
			(len(s) > len(substr) && indexOf(s, substr) >= 0))
}

func indexOf(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
