package extproc

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLoadClassifierMappingsUsesProjectedDefaultAndSkipsNamedFallback(t *testing.T) {
	cfg := newCoreSignalMappingGateConfig(t)
	root := t.TempDir()
	mapping := filepath.Join(root, "labels.json")
	require.NoError(t, os.WriteFile(mapping, []byte(`{"category_to_idx":{"billing":0,"chat":1},"idx_to_category":{"0":"billing","1":"chat"}}`), 0o600))
	cfg.Decisions = []config.Decision{{Name: "route", Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"}}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"replacement": {Provider: "candle", Artifact: root, Device: "cpu"}}
	cfg.ModelBindings = map[string]config.ModelBinding{"domain_classifier": {Deployment: "replacement", Adapter: "auto", Contract: config.RemoteClassifierContractLabelDistribution, MappingPath: mapping}}
	mappings, err := loadClassifierMappings(cfg)
	require.NoError(t, err)
	require.Equal(t, 0, mappings.categoryMapping.CategoryToIdx["billing"])
	require.NotEqual(t, mapping, cfg.CategoryMappingPath)
	cfg.RouterOptions.AutoModelNames = []string{}
	cfg.Recipes = []config.RoutingRecipe{{Name: config.DefaultRecipeName}, {Name: "named", Profile: config.RoutingProfile{Decisions: cfg.Decisions, ModelBindings: cfg.ModelBindings}}}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"named-entry"}, Recipe: "named"}}
	mappings, err = loadClassifierMappings(cfg)
	require.NoError(t, err)
	require.Nil(t, mappings.categoryMapping)
}

func TestLoadClassifierMappingsSkipsUnusedCoreSignals(t *testing.T) {
	cfg := newCoreSignalMappingGateConfig(t)

	mappings, err := loadClassifierMappings(cfg)
	require.NoError(t, err)
	require.NotNil(t, mappings)
	require.Nil(t, mappings.categoryMapping)
	require.Nil(t, mappings.piiMapping)
	require.Nil(t, mappings.jailbreakMapping)

	components, err := buildRouterComponents(cfg)
	require.NoError(t, err)
	require.NotNil(t, components)
	require.NotNil(t, components.classifier)
	require.NotNil(t, components.classificationSvc)
}

func TestLoadClassifierMappingsRequiresUsedCoreSignalMappings(t *testing.T) {
	tests := []struct {
		name        string
		rule        config.RuleNode
		wantErrPart string
	}{
		{
			name:        "domain signal",
			rule:        config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"},
			wantErrPart: "failed to load category mapping",
		},
		{
			name:        "pii signal",
			rule:        config.RuleNode{Type: config.SignalTypePII, Name: "contains_pii"},
			wantErrPart: "failed to load PII mapping",
		},
		{
			name:        "jailbreak signal",
			rule:        config.RuleNode{Type: config.SignalTypeJailbreak, Name: "detector"},
			wantErrPart: "failed to load jailbreak mapping",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := newCoreSignalMappingGateConfig(t)
			cfg.Decisions = []config.Decision{{
				Name: "guarded-route",
				Rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
					tt.rule,
				}},
			}}

			_, err := loadClassifierMappings(cfg)
			require.Error(t, err)
			require.Contains(t, err.Error(), tt.wantErrPart)
		})
	}
}

func TestBuildRouterComponentsClosesEarlierResourcesOnLaterFailure(t *testing.T) {
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{
			Enabled:    true,
			TTLSeconds: 60,
		},
		InlineModels: config.InlineModels{
			PromptGuard: config.PromptGuardConfig{
				Enabled:  true,
				Protocol: config.PromptGuardProtocolHTTPClassify,
			},
		},
	}

	baseline := stableGoroutineCount(t)

	components, err := buildRouterComponents(cfg)
	require.Error(t, err)
	require.Nil(t, components)

	require.Eventually(t, func() bool {
		runtime.GC()
		return runtime.NumGoroutine() <= baseline
	}, 10*time.Second, 10*time.Millisecond)
}

func stableGoroutineCount(t *testing.T) int {
	t.Helper()
	var last int
	consecutive := 0
	require.Eventually(t, func() bool {
		runtime.GC()
		current := runtime.NumGoroutine()
		if current == last {
			consecutive++
		} else {
			consecutive = 0
			last = current
		}
		return consecutive >= 3
	}, 10*time.Second, 10*time.Millisecond, "goroutine count never settled")
	return last
}

func newCoreSignalMappingGateConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	missingRoot := filepath.Join(t.TempDir(), "missing-model-assets")
	return &config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "models/mmbert32k-intent-classifier-merged",
					CategoryMappingPath: filepath.Join(missingRoot, "category_mapping.json"),
				},
				PIIModel: config.PIIModel{
					ModelID:        "models/mmbert32k-pii-detector-merged",
					PIIMappingPath: filepath.Join(missingRoot, "pii_type_mapping.json"),
				},
			},
			PromptGuard: config.PromptGuardConfig{
				Enabled:              true,
				ModelID:              "models/mmbert32k-jailbreak-detector-merged",
				JailbreakMappingPath: filepath.Join(missingRoot, "jailbreak_type_mapping.json"),
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:  "default-route",
				Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{}},
			}},
		},
	}
}
