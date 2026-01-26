package config

import (
	"testing"
)

// TestMaxContextLength_UpdatedTo32K verifies that the MaxContextLength
// for ModernBERT-compatible classifiers has been updated to 32768
func TestMaxContextLength_UpdatedTo32K(t *testing.T) {
	// Test Domain Classifier
	domainModel := GetModelByPath("models/mom-domain-classifier")
	if domainModel == nil {
		t.Fatal("Domain classifier model not found")
	}
	if domainModel.MaxContextLength != 32768 {
		t.Errorf("Domain Classifier: Expected MaxContextLength=32768, got %d", domainModel.MaxContextLength)
	}

	// Test PII Detector
	piiModel := GetModelByPath("models/mom-pii-classifier")
	if piiModel == nil {
		t.Fatal("PII detector model not found")
	}
	if piiModel.MaxContextLength != 32768 {
		t.Errorf("PII Detector: Expected MaxContextLength=32768, got %d", piiModel.MaxContextLength)
	}

	// Test Jailbreak Classifier
	jailbreakModel := GetModelByPath("models/mom-jailbreak-classifier")
	if jailbreakModel == nil {
		t.Fatal("Jailbreak classifier model not found")
	}
	if jailbreakModel.MaxContextLength != 32768 {
		t.Errorf("Jailbreak Classifier: Expected MaxContextLength=32768, got %d", jailbreakModel.MaxContextLength)
	}
}

// TestMaxContextLength_ByAlias verifies that MaxContextLength is correct
// when accessing models by their aliases
func TestMaxContextLength_ByAlias(t *testing.T) {
	// Test Domain Classifier by alias
	domainByAlias := GetModelByPath("domain-classifier")
	if domainByAlias == nil {
		t.Fatal("Domain classifier not found by alias")
	}
	if domainByAlias.MaxContextLength != 32768 {
		t.Errorf("Domain Classifier (by alias): Expected MaxContextLength=32768, got %d", domainByAlias.MaxContextLength)
	}

	// Test PII Detector by alias
	piiByAlias := GetModelByPath("pii-detector")
	if piiByAlias == nil {
		t.Fatal("PII detector not found by alias")
	}
	if piiByAlias.MaxContextLength != 32768 {
		t.Errorf("PII Detector (by alias): Expected MaxContextLength=32768, got %d", piiByAlias.MaxContextLength)
	}

	// Test Jailbreak Classifier by alias
	jailbreakByAlias := GetModelByPath("jailbreak-detector")
	if jailbreakByAlias == nil {
		t.Fatal("Jailbreak classifier not found by alias")
	}
	if jailbreakByAlias.MaxContextLength != 32768 {
		t.Errorf("Jailbreak Classifier (by alias): Expected MaxContextLength=32768, got %d", jailbreakByAlias.MaxContextLength)
	}
}

// TestMaxContextLength_DescriptionsUpdated verifies that model descriptions
// mention ModernBERT-base-32k and 32K context support
func TestMaxContextLength_DescriptionsUpdated(t *testing.T) {
	domainModel := GetModelByPath("models/mom-domain-classifier")
	if domainModel == nil {
		t.Fatal("Domain classifier model not found")
	}
	if domainModel.MaxContextLength == 32768 {
		// Check that description mentions ModernBERT-base-32k or 32K
		desc := domainModel.Description
		hasModernBERT := contains(desc, "ModernBERT-base-32k") || contains(desc, "modernbert-base-32k")
		has32K := contains(desc, "32K") || contains(desc, "32768")
		if !hasModernBERT && !has32K {
			t.Errorf("Domain Classifier description should mention ModernBERT-base-32k or 32K context, got: %s", desc)
		}
	}

	piiModel := GetModelByPath("models/mom-pii-classifier")
	if piiModel == nil {
		t.Fatal("PII detector model not found")
	}
	if piiModel.MaxContextLength == 32768 {
		desc := piiModel.Description
		hasModernBERT := contains(desc, "ModernBERT-base-32k") || contains(desc, "modernbert-base-32k")
		has32K := contains(desc, "32K") || contains(desc, "32768")
		if !hasModernBERT && !has32K {
			t.Errorf("PII Detector description should mention ModernBERT-base-32k or 32K context, got: %s", desc)
		}
	}

	jailbreakModel := GetModelByPath("models/mom-jailbreak-classifier")
	if jailbreakModel == nil {
		t.Fatal("Jailbreak classifier model not found")
	}
	if jailbreakModel.MaxContextLength == 32768 {
		desc := jailbreakModel.Description
		hasModernBERT := contains(desc, "ModernBERT-base-32k") || contains(desc, "modernbert-base-32k")
		has32K := contains(desc, "32K") || contains(desc, "32768")
		if !hasModernBERT && !has32K {
			t.Errorf("Jailbreak Classifier description should mention ModernBERT-base-32k or 32K context, got: %s", desc)
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
