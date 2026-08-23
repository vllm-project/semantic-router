package extproc

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestLoadClassifierMappingsSkipsUnusedCoreSignals(t *testing.T) {
	cfg := newCoreSignalMappingGateConfig(t)

	mappings, err := loadClassifierMappings(cfg)
	require.NoError(t, err)
	require.NotNil(t, mappings)
	require.Nil(t, mappings.categoryMapping)
	require.Nil(t, mappings.piiMapping)
	require.Nil(t, mappings.jailbreakMapping)

	components, err := buildRouterComponentsWithDependencies(cfg, RuntimeDependencies{
		DispatchCapabilities: dispatchCapabilityRuntimeStub{},
		ResponseTerminals:    backendinvoker.NewLocalResponseTerminalStore(),
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
	})
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
			cfg.Recipes[0].Profile.Decisions = []config.Decision{{
				ID: "dec-guarded", Name: "guarded-route",
				Rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
					tt.rule,
				}},
			}}
			require.NoError(t, cfg.PrepareEntrypointRecipes())

			_, err := loadClassifierMappings(cfg)
			require.Error(t, err)
			require.Contains(t, err.Error(), tt.wantErrPart)
		})
	}
}

func newCoreSignalMappingGateConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	missingRoot := filepath.Join(t.TempDir(), "missing-model-assets")
	cfg := &config.RouterConfig{
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
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"backend": {ResourceID: "mdl-backend", ResourceRevision: 1},
		}},
		Recipes: []config.RoutingRecipe{{
			ID: "rcp-default", Revision: 1, Name: config.DefaultRecipeName,
			Profile: config.RoutingProfile{Decisions: []config.Decision{{
				ID: "dec-guarded", Name: "default-route",
				Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{}},
			}}},
		}},
		Entrypoints: []config.EntrypointMapping{{
			ID: "ep-default", Revision: 1, Name: "default", ModelNames: []string{"router/default"},
			Rules: []config.EntrypointRule{{
				ID: "rule-default", Name: "default",
				Action: config.EntrypointRuleAction{
					RecipeID: "rcp-default", RecipeRevision: 1, Recipe: config.DefaultRecipeName,
					Assignments: map[string]config.RoutingAssignmentSet{
						"dec-guarded": {Models: []config.RoutingModelAssignment{{
							ModelID: "mdl-backend", ModelRevision: 1, ModelName: "backend", Weight: "1",
						}}},
					},
				},
			}},
		}},
	}
	require.NoError(t, cfg.PrepareEntrypointRecipes())
	return cfg
}

func TestCreateRouterClassifierAllowsEmptyManagedSnapshot(t *testing.T) {
	classifiers, defaultClassifier, service, err := createRouterClassifier(
		&config.RouterConfig{},
		&classifierMappings{},
	)
	require.NoError(t, err)
	require.NotNil(t, classifiers)
	require.Nil(t, defaultClassifier)
	require.NotNil(t, service)
}
