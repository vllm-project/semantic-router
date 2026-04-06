package services

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestResolveUnifiedModelsPath(t *testing.T) {
	t.Run("defaults to local models dir", func(t *testing.T) {
		require.Equal(t, "./models", resolveUnifiedModelsPath(nil))
		require.Equal(t, "./models", resolveUnifiedModelsPath(&config.RouterConfig{}))
	})

	t.Run("uses directory from configured model id", func(t *testing.T) {
		routerConfig := &config.RouterConfig{
			InlineModels: config.InlineModels{
				Classifier: config.Classifier{
					CategoryModel: config.CategoryModel{ModelID: "models/mom-domain-classifier"},
				},
			},
		}
		require.Equal(t, "models", resolveUnifiedModelsPath(routerConfig))
	})
}

func TestUseMCPCategories(t *testing.T) {
	require.True(t, useMCPCategories(&config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				MCPCategoryModel: config.MCPCategoryModel{Enabled: true},
			},
		},
	}))
	require.False(t, useMCPCategories(&config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel:    config.CategoryModel{ModelID: "models/domain"},
				MCPCategoryModel: config.MCPCategoryModel{Enabled: true},
			},
		},
	}))
	require.False(t, useMCPCategories(nil))
}

func TestLoadLegacyClassifierMappings(t *testing.T) {
	dir := t.TempDir()
	categoryPath := writeBootstrapFixture(
		t,
		dir,
		"categories.json",
		`{"category_to_idx":{"math":0},"idx_to_category":{"0":"math"}}`,
	)
	piiPath := writeBootstrapFixture(
		t,
		dir,
		"pii.json",
		`{"label_to_idx":{"PERSON":0},"idx_to_label":{"0":"PERSON"}}`,
	)
	jailbreakPath := writeBootstrapFixture(
		t,
		dir,
		"jailbreak.json",
		`{"label_to_idx":{"safe":0,"jailbreak":1},"idx_to_label":{"0":"safe","1":"jailbreak"}}`,
	)

	routerConfig := &config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{CategoryMappingPath: categoryPath},
				PIIModel:      config.PIIModel{PIIMappingPath: piiPath},
			},
			PromptGuard: config.PromptGuardConfig{
				JailbreakMappingPath: jailbreakPath,
			},
		},
	}

	mappings, err := loadLegacyClassifierMappings(routerConfig)
	require.NoError(t, err)
	require.NotNil(t, mappings.category)
	require.NotNil(t, mappings.pii)
	require.NotNil(t, mappings.jailbreak)

	categoryName, ok := mappings.category.GetCategoryFromIndex(0)
	require.True(t, ok)
	require.Equal(t, "math", categoryName)
	piiType, ok := mappings.pii.GetPIITypeFromIndex(0)
	require.True(t, ok)
	require.Equal(t, "PERSON", piiType)
	jailbreakType, ok := mappings.jailbreak.GetJailbreakTypeFromIndex(1)
	require.True(t, ok)
	require.Equal(t, "jailbreak", jailbreakType)
}

func TestLoadLegacyClassifierMappingsUsesMCPCategoryMode(t *testing.T) {
	routerConfig := &config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				MCPCategoryModel: config.MCPCategoryModel{Enabled: true},
			},
		},
	}

	mappings, err := loadLegacyClassifierMappings(routerConfig)
	require.NoError(t, err)
	require.Nil(t, mappings.category)
}

func TestCreateLegacyClassifierRejectsNilConfig(t *testing.T) {
	_, err := createLegacyClassifier(nil)
	require.EqualError(t, err, "config is nil")
}

func writeBootstrapFixture(t *testing.T, dir string, name string, contents string) string {
	t.Helper()

	path := filepath.Join(dir, name)
	require.NoError(t, os.WriteFile(path, []byte(contents), 0o600))
	return path
}
