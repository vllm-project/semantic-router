package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// validateLossyTranslationContracts rejects entries the codec cannot opt into
// instead of ignoring them, so the list states exactly what the runtime allows.
// An entrypoint's own list is held to the same entries, and only a Chat
// entrypoint has one: a systemone entrypoint translates no wire format.
func validateLossyTranslationContracts(cfg *RouterConfig) error {
	if err := validateLossyTranslationList("global.router.allow_lossy_translation", cfg.AllowLossyTranslation); err != nil {
		return err
	}
	for index, entrypoint := range cfg.Entrypoints {
		if entrypoint.AllowLossyTranslation == nil {
			continue
		}
		path := fmt.Sprintf("entrypoints[%d].allow_lossy_translation", index)
		if entrypoint.API == SystemOneAPI {
			return fmt.Errorf("%s: a systemone entrypoint translates no wire format", path)
		}
		if err := validateLossyTranslationList(path, *entrypoint.AllowLossyTranslation); err != nil {
			return err
		}
	}
	return nil
}

func validateLossyTranslationList(path string, fields []string) error {
	allowed := llmprotocol.OptInLossyFields()
	for _, field := range fields {
		if !slices.Contains(allowed, field) {
			return fmt.Errorf("%s: %q is not supported, use one of: %s",
				path, field, strings.Join(allowed, ", "))
		}
	}
	return nil
}
