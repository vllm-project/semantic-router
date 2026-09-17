package llmprotocol

import (
	"strings"
	"unicode/utf8"
)

func validateCacheSalt(salt *string) error {
	if salt != nil && (*salt == "" || utf8.RuneCountInString(*salt) > 128 || strings.ContainsAny(*salt, "@/\\\x00")) {
		return NewError(ErrorInvalidRequest, "invalid_cache_salt", "cache_salt must contain 1 to 128 characters without @, slash, backslash, or NUL", nil)
	}
	return nil
}
