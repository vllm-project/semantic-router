// Package managementsearch defines the bounded, canonical prefix-search
// contract shared by Management API collection services and PostgreSQL stores.
package managementsearch

import (
	"errors"
	"strings"
	"unicode"
	"unicode/utf8"
)

const MaximumRunes = 200

var ErrInvalid = errors.New("management search is invalid")

// Normalize returns the canonical value that is bound into opaque cursors.
// Empty and whitespace-only searches intentionally mean "no search".
func Normalize(value string) (string, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	if !utf8.ValidString(value) || utf8.RuneCountInString(value) > MaximumRunes {
		return "", ErrInvalid
	}
	for _, character := range value {
		if unicode.IsControl(character) {
			return "", ErrInvalid
		}
	}
	return value, nil
}

// PrefixPattern escapes SQL LIKE metacharacters and returns a literal prefix
// pattern suitable for expression indexes using text_pattern_ops.
func PrefixPattern(normalized string) string {
	if normalized == "" {
		return ""
	}
	replacer := strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`)
	return replacer.Replace(normalized) + "%"
}
