package classification

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestNewLegacyClassifierFromConfigRejectsNilConfig(t *testing.T) {
	classifier, err := NewLegacyClassifierFromConfig(nil)
	require.Nil(t, classifier)
	require.Error(t, err)
	require.Contains(t, err.Error(), "config is nil")
}

func TestNewLegacyClassifierFromConfigSkipsUnusedCoreSignalMappings(t *testing.T) {
	cfg := newLegacyClassifierMappingGateConfig(t)

	classifier, err := NewLegacyClassifierFromConfig(cfg)
	require.NoError(t, err)
	require.NotNil(t, classifier)
}

func TestNewLegacyClassifierFromConfigRequiresUsedCoreSignalMappings(t *testing.T) {
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
			cfg := newLegacyClassifierMappingGateConfig(t)
			cfg.Decisions = []config.Decision{{
				Name: "guarded-route",
				Rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
					tt.rule,
				}},
			}}

			classifier, err := NewLegacyClassifierFromConfig(cfg)
			require.Nil(t, classifier)
			require.Error(t, err)
			require.Contains(t, err.Error(), tt.wantErrPart)
		})
	}
}

func newLegacyClassifierMappingGateConfig(t *testing.T) *config.RouterConfig {
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
