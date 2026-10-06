package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// NewLegacyClassifierFromConfig loads mapping assets and builds the legacy
// classifier runtime for callers that still use the non-unified path.
func NewLegacyClassifierFromConfig(cfg *config.RouterConfig, runtimeOptions ...RecipeRuntimeOptions) (*Classifier, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is nil")
	}
	classifier, err := buildClassifierWithAdmission(cfg, nil, nil, nil, nil, runtimeOptions...)
	if err != nil {
		return nil, fmt.Errorf("failed to create classifier: %w", err)
	}
	if err := classifier.InitializeRuntime(); err != nil {
		_ = classifier.Close()
		return nil, fmt.Errorf("failed to initialize classifier: %w", err)
	}
	return classifier, nil
}
