package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (b *classifierOptionBuilder) buildKeywordClassifierOption() (option, error) {
	if len(b.cfg.KeywordRules) == 0 {
		return nil, nil
	}
	keywordClassifier, err := NewKeywordClassifier(b.cfg.KeywordRules)
	if err != nil {
		logging.ComponentErrorEvent("classifier", "keyword_classifier_create_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return nil, err
	}
	return withKeywordClassifier(keywordClassifier), nil
}

func (b *classifierOptionBuilder) buildContextClassifierOption() (option, error) {
	if len(b.cfg.ContextRules) == 0 {
		return nil, nil
	}
	tokenCounter := NewCalibratedTokenCounter(WithConservativeEstimate())
	return withCalibratedContextClassifier(NewContextClassifier(tokenCounter, b.cfg.ContextRules), tokenCounter), nil
}

func (b *classifierOptionBuilder) buildAuthzClassifierOption() (option, error) {
	roleBindings := b.cfg.GetRoleBindings()
	if len(roleBindings) == 0 {
		return nil, nil
	}
	authzClassifier, err := NewAuthzClassifier(roleBindings)
	if err != nil {
		return nil, fmt.Errorf("failed to create authz classifier: %w", err)
	}
	logging.ComponentEvent("classifier", "authz_classifier_initialized", map[string]interface{}{
		"role_bindings": len(roleBindings),
	})
	return withAuthzClassifier(authzClassifier), nil
}
