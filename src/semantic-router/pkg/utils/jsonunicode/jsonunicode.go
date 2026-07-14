// Package jsonunicode validates the Unicode representation of raw JSON before
// encoding/json can apply lossy replacement.
package jsonunicode

import (
	"bytes"
	"unicode/utf8"
)

// Valid reports whether raw contains valid UTF-8 and every JSON Unicode escape
// inside a string uses paired UTF-16 surrogates. It intentionally does not
// validate JSON syntax; callers must still decode the document afterward.
//
// This preflight closes two encoding/json behaviors that are unsafe for
// identity- and inference-sensitive strings: invalid UTF-8 bytes and unpaired
// surrogate escapes are otherwise replaced with U+FFFD without an error.
func Valid(raw []byte) bool {
	if !utf8.Valid(raw) {
		return false
	}

	for cursor := 0; cursor < len(raw); {
		quote := bytes.IndexByte(raw[cursor:], '"')
		if quote < 0 {
			return true
		}
		next, ok := scanJSONString(raw, cursor+quote+1)
		if !ok {
			return false
		}
		cursor = next
	}
	return true
}

func scanJSONString(raw []byte, cursor int) (int, bool) {
	for cursor < len(raw) {
		switch raw[cursor] {
		case '"':
			return cursor + 1, true
		case '\\':
			next, ok := consumeJSONStringEscape(raw, cursor)
			if !ok {
				return 0, false
			}
			cursor = next
		default:
			cursor++
		}
	}
	// JSON syntax validation reports an unterminated string after this Unicode
	// preflight; no lossy conversion can occur without another escape here.
	return len(raw), true
}

func consumeJSONStringEscape(raw []byte, slashIndex int) (int, bool) {
	if slashIndex+1 >= len(raw) {
		return 0, false
	}
	if raw[slashIndex+1] != 'u' {
		return slashIndex + 2, true
	}
	codeUnit, ok := decodeUnicodeEscape(raw, slashIndex)
	if !ok || isLowSurrogate(codeUnit) {
		return 0, false
	}
	if !isHighSurrogate(codeUnit) {
		return slashIndex + 6, true
	}
	low, ok := decodeUnicodeEscape(raw, slashIndex+6)
	if !ok || !isLowSurrogate(low) {
		return 0, false
	}
	return slashIndex + 12, true
}

func isHighSurrogate(codeUnit uint16) bool {
	return codeUnit >= 0xD800 && codeUnit <= 0xDBFF
}

func isLowSurrogate(codeUnit uint16) bool {
	return codeUnit >= 0xDC00 && codeUnit <= 0xDFFF
}

func decodeUnicodeEscape(raw []byte, slashIndex int) (uint16, bool) {
	if slashIndex < 0 || slashIndex+6 > len(raw) ||
		raw[slashIndex] != '\\' || raw[slashIndex+1] != 'u' {
		return 0, false
	}
	var value uint16
	for _, digit := range raw[slashIndex+2 : slashIndex+6] {
		nibble, ok := hexNibble(digit)
		if !ok {
			return 0, false
		}
		value = value<<4 | uint16(nibble)
	}
	return value, true
}

func hexNibble(digit byte) (byte, bool) {
	switch {
	case digit >= '0' && digit <= '9':
		return digit - '0', true
	case digit >= 'a' && digit <= 'f':
		return digit - 'a' + 10, true
	case digit >= 'A' && digit <= 'F':
		return digit - 'A' + 10, true
	default:
		return 0, false
	}
}
