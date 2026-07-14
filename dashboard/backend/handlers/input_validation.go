package handlers

import (
	"unicode"
	"unicode/utf8"
)

func containsUnicodeControl(value string) bool {
	if !utf8.ValidString(value) {
		return true
	}
	for _, r := range value {
		if unicode.IsControl(r) || unicode.In(r, unicode.Cf) || r == '\u2028' || r == '\u2029' {
			return true
		}
	}
	return false
}
