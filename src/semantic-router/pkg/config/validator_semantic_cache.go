package config

import (
	"fmt"
	"strings"
)

func validateCacheMode(mode string, scope string) error {
	switch strings.TrimSpace(mode) {
	case "", ResponseCacheModeSemantic, ResponseCacheModeExact, ResponseCacheModeExactThenSemantic:
		return nil
	default:
		return fmt.Errorf(
			"%s mode must be one of %q, %q, or %q, got %q",
			scope,
			ResponseCacheModeSemantic,
			ResponseCacheModeExact,
			ResponseCacheModeExactThenSemantic,
			mode,
		)
	}
}
