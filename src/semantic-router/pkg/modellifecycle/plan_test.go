package modellifecycle

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildPlanUsesCanonicalMmBERTDefaultWithoutLegacyBERT(t *testing.T) {
	cfg := config.DefaultGlobalConfig()

	plan := BuildPlan(&cfg)
	paths := plan.EmbeddingPaths()

	if paths.MmBERT != "models/mmbert-embed-32k-2d-matryoshka" {
		t.Fatalf("MmBERT path = %q", paths.MmBERT)
	}
	if paths.BERT != "" {
		t.Fatalf("BERT path = %q, want empty when canonical mmBERT is configured", paths.BERT)
	}
	assertDownloadPaths(t, plan, "models/mmbert-embed-32k-2d-matryoshka")
}

func TestBuildPlanAddsManagedBERTFallbackWhenBERTIsRequired(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{UseCPU: true},
		},
		SemanticCache: config.SemanticCache{
			Enabled:        true,
			EmbeddingModel: "bert",
		},
		MoMRegistry: config.ToLegacyRegistry(),
	}

	plan := BuildPlan(cfg)
	paths := plan.EmbeddingPaths()

	if paths.BERT != DefaultBERTEmbeddingModelPath {
		t.Fatalf("BERT path = %q, want %q", paths.BERT, DefaultBERTEmbeddingModelPath)
	}
	assertDownloadPaths(t, plan, DefaultBERTEmbeddingModelPath)
}

func TestBuildPlanEmbeddingPathsExposeUseCPUPolicy(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath: "models/mmbert-embed-32k-2d-matryoshka",
				UseCPU:          false,
			},
		},
	}

	paths := BuildPlan(cfg).EmbeddingPaths()
	if paths.MmBERT == "" {
		t.Fatal("MmBERT path is empty")
	}
	if paths.UseCPU {
		t.Fatal("EmbeddingPaths.UseCPU = true, want false from lifecycle plan")
	}
}

func TestConfiguredAssetsCarryLifecycleMetadata(t *testing.T) {
	cfg := config.DefaultGlobalConfig()

	asset, ok := BuildPlan(&cfg).AssetForRole(RoleMmBERTEmbedding)
	if !ok {
		t.Fatal("missing mmBERT embedding asset")
	}
	if asset.Kind != config.ModelLifecycleKindEmbedding {
		t.Fatalf("asset kind = %q", asset.Kind)
	}
	if asset.RuntimeName != "mmbert" {
		t.Fatalf("runtime name = %q", asset.RuntimeName)
	}
	if asset.DownloadTiming != config.ModelLifecycleDownloadOnUse {
		t.Fatalf("download timing = %q", asset.DownloadTiming)
	}
	if asset.InitializationTiming != config.ModelLifecycleInitRouterStartup {
		t.Fatalf("init timing = %q", asset.InitializationTiming)
	}
}

func TestBuildPlanGatesClassifierDownloadsByRoutingUsage(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "models/mmbert32k-intent-classifier-merged",
					CategoryMappingPath: "models/mmbert32k-intent-classifier-merged/category_mapping.json",
				},
			},
		},
	}

	plan := BuildPlan(cfg)
	if len(plan.DownloadAssets()) != 0 {
		t.Fatalf("DownloadAssets() = %#v, want no unused classifier assets", plan.DownloadAssets())
	}
	if got := plan.ConfiguredModelPaths(); !reflect.DeepEqual(got, []string{"models/mmbert32k-intent-classifier-merged"}) {
		t.Fatalf("ConfiguredModelPaths() = %#v", got)
	}
}

func TestBuildPlanIncludesClassifierWhenSignalIsUsed(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: config.ToLegacyRegistry(),
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "models/mmbert32k-intent-classifier-merged",
					CategoryMappingPath: "models/mmbert32k-intent-classifier-merged/category_mapping.json",
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:  "domain-route",
				Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"},
			}},
		},
	}

	plan := BuildPlan(cfg)
	assertDownloadPaths(t, plan, "models/mmbert32k-intent-classifier-merged")
	files := plan.RequiredFilesByModel()
	want := []string{"category_mapping.json"}
	if !reflect.DeepEqual(files["models/mmbert32k-intent-classifier-merged"], want) {
		t.Fatalf("required files = %#v, want %#v", files["models/mmbert32k-intent-classifier-merged"], want)
	}
}

func TestBalanceRecipeLifecyclePlanDownloadsOnlyActivatedBuiltIns(t *testing.T) {
	cfg, err := config.Parse(filepath.Join("..", "..", "..", "..", "deploy", "recipes", "balance.yaml"))
	if err != nil {
		t.Fatalf("parse balance recipe: %v", err)
	}

	plan := BuildPlan(cfg)
	assertDownloadPaths(t, plan,
		config.DefaultModelPathForLifecycleRole(config.ModelLifecycleRoleMmBERTEmbedding),
		config.DefaultModelPathForLifecycleRole(config.ModelLifecycleRoleDomainClassifier),
		config.DefaultModelPathForLifecycleRole(config.ModelLifecycleRoleFactCheckClassifier),
		config.DefaultModelPathForLifecycleRole(config.ModelLifecycleRoleFeedbackDetector),
	)
	if _, ok := plan.AssetForRole(RoleBERTEmbedding); ok {
		t.Fatal("balance recipe should not require the legacy BERT embedding fallback")
	}
}

func TestResolveMemoryEmbeddingModelPrefersConfiguredCanonicalDefault(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	if got := ResolveMemoryEmbeddingModel(&cfg); got != "mmbert" {
		t.Fatalf("ResolveMemoryEmbeddingModel() = %q, want mmbert", got)
	}
}

func TestResolveSemanticCacheEmbeddingModelDoesNotInferFromModelPaths(t *testing.T) {
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath: "models/mmbert-embed-32k-2d-matryoshka",
			},
		},
	}

	if got := ResolveSemanticCacheEmbeddingModel(cfg); got != "bert" {
		t.Fatalf("ResolveSemanticCacheEmbeddingModel() = %q, want bert fallback", got)
	}
}

func TestResolveMemoryEmbeddingModelDoesNotInferFromModelPaths(t *testing.T) {
	cfg := &config.RouterConfig{
		Memory: config.MemoryConfig{Enabled: true},
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath: "models/mmbert-embed-32k-2d-matryoshka",
			},
		},
	}

	if got := ResolveMemoryEmbeddingModel(cfg); got != "bert" {
		t.Fatalf("ResolveMemoryEmbeddingModel() = %q, want bert fallback", got)
	}
}

func assertDownloadPaths(t *testing.T, plan Plan, want ...string) {
	t.Helper()

	assets := plan.DownloadAssets()
	got := make([]string, 0, len(assets))
	for _, asset := range assets {
		got = append(got, asset.LocalPath)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("download paths = %#v, want %#v", got, want)
	}
}
