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

	if catalog.Embeddings.Semantic.ModelRefs.MmBERT != string(ModelLifecycleRoleMmBERTEmbedding) {
		t.Fatalf("semantic mmbert model_ref = %q", catalog.Embeddings.Semantic.ModelRefs.MmBERT)
	}
	runtimeDefaults := DefaultGlobalConfig()
	if runtimeDefaults.EmbeddingModels.MmBertModelPath != DefaultModelPathForLifecycleRole(ModelLifecycleRoleMmBERTEmbedding) {
		t.Fatalf("semantic mmbert runtime default = %q", runtimeDefaults.EmbeddingModels.MmBertModelPath)
	}
	if runtimeDefaults.EmbeddingModels.BertModelPath != "" {
		t.Fatalf("canonical semantic runtime default should not configure BERT fallback, got %q", runtimeDefaults.EmbeddingModels.BertModelPath)
	}

	system := catalog.System
	for _, binding := range DefaultModelLifecycleBindings() {
		assertDefaultSystemBinding(t, binding.Role, system.ModelPath(binding.Role))
	}
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

func TestCanonicalExportUsesLifecycleCatalogRoles(t *testing.T) {
	cfg := DefaultGlobalConfig()
	cfg.EmbeddingModels.Qwen3ModelPath = "models/custom-qwen3-embedding"
	cfg.ModalityDetector.Classifier = &ModalityClassifierConfig{ModelPath: "models/custom-modality"}

	global := CanonicalGlobalFromRouterConfig(&cfg)
	if global == nil {
		t.Fatal("expected canonical global export")
	}

	for _, binding := range DefaultModelLifecycleBindings() {
		want := routerConfigModelPathForLifecycleRole(&cfg, binding.Role)
		if want == "" {
			continue
		}
		if got := global.ModelCatalog.System.ModelPath(binding.Role); got != want {
			t.Fatalf("%s exported system binding = %q, want %q", binding.Role, got, want)
		}
	}
	if got := global.ModelCatalog.Embeddings.Semantic.ModelRefs.Qwen3; got != string(ModelLifecycleRoleQwen3Embedding) {
		t.Fatalf("exported qwen3 embedding ref = %q", got)
	}
	if global.ModelCatalog.Modules.ModalityDetector.Classifier == nil {
		t.Fatal("expected exported modality classifier")
	}
	if got := global.ModelCatalog.Modules.ModalityDetector.Classifier.ModelRef; got != string(ModelLifecycleRoleModalityClassifier) {
		t.Fatalf("exported modality classifier ref = %q", got)
	}
}

func assertDefaultSystemBinding(t *testing.T, role ModelLifecycleRole, got string) {
	t.Helper()
	if want := DefaultModelPathForLifecycleRole(role); got != want {
		t.Fatalf("%s default = %q, want %q", role, got, want)
	}
}
