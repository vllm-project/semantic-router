package systemoneprotocol

import (
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Canonical is the value as Python's json.dumps(value, ensure_ascii=False)
// writes it, with the default separators ", " and ": " and members in their
// order:
//   - strings escape '"', '\\', \n, \r, \t, \b and \f by name and every other
//     control character below U+0020 as \u00xx, and keep everything else as
//     is: non-ASCII, U+2028 and U+2029 unescaped, and no HTML escaping;
//   - an integer is written as Python's int writes it, and any other number
//     as Python's float repr writes it (1e16 as 1e+16, 0.00001 as 1e-05,
//     100 as 100.0), Infinity past the float range.
//
// The vllm-sr families render an object state in exactly this form, so the
// text of an object reads as the object itself.
func Canonical(v Value) string {
	return string(appendValue(nil, v, true))
}

// CanonicalJSON is Canonical of the JSON value data holds.
func CanonicalJSON(data []byte) (string, error) {
	value, err := Parse(data)
	if err != nil {
		return "", err
	}
	return Canonical(value), nil
}

func appendValue(dst []byte, v Value, canonical bool) []byte {
	switch v.kind {
	case KindBool:
		if v.boolean {
			return append(dst, "true"...)
		}
		return append(dst, "false"...)
	case KindNumber:
		if canonical {
			return appendPythonNumber(dst, v.text)
		}
		return append(dst, v.text...)
	case KindString:
		return appendPythonString(dst, v.text)
	case KindArray:
		dst = append(dst, '[')
		for index, item := range v.items {
			if index > 0 {
				dst = appendSeparator(dst, ',', canonical)
			}
			dst = appendValue(dst, item, canonical)
		}
		return append(dst, ']')
	case KindObject:
		dst = append(dst, '{')
		for index, member := range v.members {
			if index > 0 {
				dst = appendSeparator(dst, ',', canonical)
			}
			dst = appendPythonString(dst, member.Key)
			dst = appendSeparator(dst, ':', canonical)
			dst = appendValue(dst, member.Value, canonical)
		}
		return append(dst, '}')
	}
	return append(dst, "null"...)
}

func appendSeparator(dst []byte, separator byte, canonical bool) []byte {
	dst = append(dst, separator)
	if canonical {
		dst = append(dst, ' ')
	}
	return dst
}

const hexDigits = "0123456789abcdef"

// appendPythonString writes s as Python's json encoder does with
// ensure_ascii=False. Invalid UTF-8 is written as U+FFFD, which is what a
// JSON reader gives for it.
func appendPythonString(dst []byte, s string) []byte {
	dst = append(dst, '"')
	for index := 0; index < len(s); {
		c := s[index]
		if c < utf8.RuneSelf {
			switch c {
			case '"':
				dst = append(dst, '\\', '"')
			case '\\':
				dst = append(dst, '\\', '\\')
			case '\n':
				dst = append(dst, '\\', 'n')
			case '\r':
				dst = append(dst, '\\', 'r')
			case '\t':
				dst = append(dst, '\\', 't')
			case '\b':
				dst = append(dst, '\\', 'b')
			case '\f':
				dst = append(dst, '\\', 'f')
			default:
				if c < 0x20 {
					dst = append(dst, '\\', 'u', '0', '0', hexDigits[c>>4], hexDigits[c&0xf])
				} else {
					dst = append(dst, c)
				}
			}
			index++
			continue
		}
		r, size := utf8.DecodeRuneInString(s[index:])
		if r == utf8.RuneError && size == 1 {
			dst = utf8.AppendRune(dst, utf8.RuneError)
		} else {
			dst = append(dst, s[index:index+size]...)
		}
		index += size
	}
	return append(dst, '"')
}

// appendPythonNumber writes a JSON number literal as Python writes the
// number json.loads reads from it: an int for a literal without a fraction
// or exponent, else a float.
func appendPythonNumber(dst []byte, literal string) []byte {
	if !strings.ContainsAny(literal, ".eE") {
		if literal == "-0" {
			return append(dst, '0')
		}
		return append(dst, literal...)
	}
	value, err := strconv.ParseFloat(literal, 64)
	if err != nil && !math.IsInf(value, 0) {
		return append(dst, literal...)
	}
	return appendPythonFloat(dst, value)
}

// appendPythonFloat writes value as Python's float repr does: the shortest
// digits that round-trip, in fixed notation when the decimal exponent is in
// [-4, 16), with ".0" for an integral value, and otherwise as d.ddde±XX.
func appendPythonFloat(dst []byte, value float64) []byte {
	switch {
	case math.IsInf(value, 1):
		return append(dst, "Infinity"...)
	case math.IsInf(value, -1):
		return append(dst, "-Infinity"...)
	case math.IsNaN(value):
		return append(dst, "NaN"...)
	}
	formatted := strconv.FormatFloat(value, 'e', -1, 64)
	if formatted[0] == '-' {
		dst = append(dst, '-')
		formatted = formatted[1:]
	}
	mantissa, exponentText, _ := strings.Cut(formatted, "e")
	exponent, _ := strconv.Atoi(exponentText)
	digits := strings.Replace(mantissa, ".", "", 1)
	if value == 0 {
		digits, exponent = "0", 0
	}
	// The value is 0.digits × 10^point.
	point := exponent + 1
	if -4 < point && point <= 16 {
		switch {
		case point <= 0:
			dst = append(dst, '0', '.')
			dst = append(dst, strings.Repeat("0", -point)...)
			return append(dst, digits...)
		case point >= len(digits):
			dst = append(dst, digits...)
			dst = append(dst, strings.Repeat("0", point-len(digits))...)
			return append(dst, '.', '0')
		default:
			dst = append(dst, digits[:point]...)
			dst = append(dst, '.')
			return append(dst, digits[point:]...)
		}
	}
	dst = append(dst, digits[0])
	if len(digits) > 1 {
		dst = append(dst, '.')
		dst = append(dst, digits[1:]...)
	}
	dst = append(dst, 'e')
	if exponent < 0 {
		dst = append(dst, '-')
		exponent = -exponent
	} else {
		dst = append(dst, '+')
	}
	if exponent < 10 {
		dst = append(dst, '0')
	}
	return strconv.AppendInt(dst, int64(exponent), 10)
}
