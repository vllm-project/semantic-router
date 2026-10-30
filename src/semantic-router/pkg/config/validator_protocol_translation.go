package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// validateLossyTranslationContracts rejects entries the codec cannot opt into
// instead of ignoring them, so the list states exactly what the runtime allows.
func validateLossyTranslationContracts(cfg *RouterConfig) error {
	allowed := llmprotocol.OptInLossyFields()
	for _, field := range cfg.AllowLossyTranslation {
		if !slices.Contains(allowed, field) {
			return fmt.Errorf("global.router.allow_lossy_translation: %q is not supported, use one of: %s",
				field, strings.Join(allowed, ", "))
		}
	}
	return nil
}
