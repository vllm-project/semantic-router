package config

import "testing"

func TestModelLifecycleBindingsResolveThroughDefaultRegistry(t *testing.T) {
	for _, binding := range DefaultModelLifecycleBindings() {
		if binding.Role == "" {
			t.Fatal("lifecycle binding has empty role")
		}
		if binding.DefaultLocalPath == "" {
			t.Fatalf("lifecycle binding %q has empty default path", binding.Role)
		}
		if GetModelByPath(binding.DefaultLocalPath) == nil {
			t.Fatalf("lifecycle binding %q default path %q is missing from DefaultModelRegistry", binding.Role, binding.DefaultLocalPath)
		}
	}
}

func TestCanonicalDefaultsUseLifecycleCatalogBindings(t *testing.T) {
	catalog := defaultCanonicalModelCatalog()

	if catalog.Embeddings.Semantic.MmBertModelPath != DefaultModelPathForLifecycleRole(ModelLifecycleRoleMmBERTEmbedding) {
		t.Fatalf("semantic mmbert default = %q", catalog.Embeddings.Semantic.MmBertModelPath)
	}
	if catalog.Embeddings.Semantic.BertModelPath != "" {
		t.Fatalf("canonical semantic default should not configure BERT fallback, got %q", catalog.Embeddings.Semantic.BertModelPath)
	}

	system := catalog.System
	assertDefaultSystemBinding(t, "prompt_guard", system.PromptGuard, ModelLifecycleRolePromptGuard)
	assertDefaultSystemBinding(t, "domain_classifier", system.DomainClassifier, ModelLifecycleRoleDomainClassifier)
	assertDefaultSystemBinding(t, "pii_classifier", system.PIIClassifier, ModelLifecycleRolePIIClassifier)
	assertDefaultSystemBinding(t, "fact_check_classifier", system.FactCheckClassifier, ModelLifecycleRoleFactCheckClassifier)
	assertDefaultSystemBinding(t, "hallucination_detector", system.HallucinationDetector, ModelLifecycleRoleHallucinationModel)
	assertDefaultSystemBinding(t, "hallucination_explainer", system.HallucinationExplainer, ModelLifecycleRoleHallucinationNLI)
	assertDefaultSystemBinding(t, "feedback_detector", system.FeedbackDetector, ModelLifecycleRoleFeedbackDetector)
}

func TestCanonicalModuleCompanionPathsUseLifecycleCatalog(t *testing.T) {
	modules := defaultCanonicalModelModules()

	if got := modules.PromptGuard.JailbreakMappingPath; got != DefaultCompanionPathForLifecycleRole(ModelLifecycleRolePromptGuard) {
		t.Fatalf("prompt guard companion path = %q", got)
	}
	if got := modules.Classifier.Domain.CategoryMappingPath; got != DefaultCompanionPathForLifecycleRole(ModelLifecycleRoleDomainClassifier) {
		t.Fatalf("domain classifier companion path = %q", got)
	}
	if got := modules.Classifier.PII.PIIMappingPath; got != DefaultCompanionPathForLifecycleRole(ModelLifecycleRolePIIClassifier) {
		t.Fatalf("PII classifier companion path = %q", got)
	}
}

func assertDefaultSystemBinding(t *testing.T, name string, got string, role ModelLifecycleRole) {
	t.Helper()
	if want := DefaultModelPathForLifecycleRole(role); got != want {
		t.Fatalf("%s default = %q, want %q", name, got, want)
	}
}
