package classification

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestNewRecipeClassifiersFromConfigRejectsNilConfig(t *testing.T) {
	classifiers, err := NewRecipeClassifiersFromConfig(nil)
	require.Nil(t, classifiers)
	require.Error(t, err)
	require.Contains(t, err.Error(), "config is nil")
}

func TestNewRecipeClassifiersFromConfigSkipsUnusedCoreSignalMappings(t *testing.T) {
	cfg := newRecipeClassifierMappingGateConfig(t, config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{}})

	classifiers, err := NewRecipeClassifiersFromConfig(cfg)
	require.NoError(t, err)
	require.NotNil(t, classifiers)
}

func TestNewRecipeClassifiersFromConfigIgnoresRootRoutingFields(t *testing.T) {
	cfg := &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
		Decisions: []config.Decision{{Name: "implicit-route"}},
	}}

	classifiers, err := NewRecipeClassifiersFromConfig(cfg)
	require.NoError(t, err)
	require.NotNil(t, classifiers)
	_, found := classifiers.ForRecipe(config.DefaultRecipeName)
	require.False(t, found)
}

func TestNewRecipeClassifiersFromConfigRequiresUsedCoreSignalMappings(t *testing.T) {
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
			cfg := newRecipeClassifierMappingGateConfig(t, config.RuleNode{
				Operator: "OR", Conditions: []config.RuleNode{tt.rule},
			})

			classifiers, err := NewRecipeClassifiersFromConfig(cfg)
			require.Nil(t, classifiers)
			require.Error(t, err)
			require.Contains(t, err.Error(), tt.wantErrPart)
		})
	}
}

func newRecipeClassifierMappingGateConfig(t *testing.T, rules config.RuleNode) *config.RouterConfig {
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
				ID: "dec-route", Name: "route", Rules: rules,
			}}},
		}},
		Entrypoints: []config.EntrypointMapping{{
			ID: "ep-default", Revision: 1, Name: "router/default", ModelNames: []string{"router/default"},
			Rules: []config.EntrypointRule{{
				ID: "rule-default", Name: "default",
				Action: config.EntrypointRuleAction{
					RecipeID: "rcp-default", RecipeRevision: 1, Recipe: config.DefaultRecipeName,
					Assignments: map[string]config.RoutingAssignmentSet{
						"dec-route": {Models: []config.RoutingModelAssignment{{
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
