package headers

import (
	"math"
	"strconv"
	"strings"
)

// Item is an RFC 8941 structured-field item or inner list, with its
// parameters, ready to serialize.
type Item struct {
	bare   string
	params string
}

const (
	maxStructuredInteger = 999_999_999_999_999
	maxStructuredDecimal = 999_999_999_999.999
)

// String is a string item. A string percent-encodes "%" and every byte
// outside printable ASCII, so any text survives the field.
func String(value string) Item {
	var out strings.Builder
	out.Grow(len(value) + 2)
	out.WriteByte('"')
	for i := 0; i < len(value); i++ {
		switch b := value[i]; {
		case b == '"' || b == '\\':
			out.WriteByte('\\')
			out.WriteByte(b)
		case b == '%' || b < 0x20 || b > 0x7e:
			out.WriteByte('%')
			out.WriteByte("0123456789ABCDEF"[b>>4])
			out.WriteByte("0123456789ABCDEF"[b&0x0f])
		default:
			out.WriteByte(b)
		}
	}
	out.WriteByte('"')
	return Item{bare: out.String()}
}

// Token is a token item, or a string item when value is no valid token.
func Token(value string) Item {
	if !isToken(value) {
		return String(value)
	}
	return Item{bare: value}
}

func isToken(value string) bool {
	if value == "" || (!isAlpha(value[0]) && value[0] != '*') {
		return false
	}
	for i := 1; i < len(value); i++ {
		b := value[i]
		if !isAlpha(b) && (b < '0' || b > '9') && !strings.ContainsRune("!#$%&'*+-.^_`|~:/", rune(b)) {
			return false
		}
	}
	return true
}

func isAlpha(b byte) bool { return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' }

// Integer is an integer item, clamped to the fifteen digits RFC 8941 allows.
func Integer(value int64) Item {
	value = max(min(value, maxStructuredInteger), -maxStructuredInteger)
	return Item{bare: strconv.FormatInt(value, 10)}
}

// Decimal is a decimal item rounded to the three fractional digits RFC 8941
// allows.
func Decimal(value float64) Item {
	if math.IsNaN(value) {
		value = 0
	}
	value = math.Round(max(min(value, maxStructuredDecimal), -maxStructuredDecimal)*1000) / 1000
	if value == 0 {
		value = 0 // rounding leaves -0 for small negative values
	}
	text := strconv.FormatFloat(value, 'f', -1, 64)
	if !strings.Contains(text, ".") {
		text += ".0"
	}
	return Item{bare: text}
}

// Boolean is a boolean item.
func Boolean(value bool) Item {
	if value {
		return Item{bare: "?1"}
	}
	return Item{bare: "?0"}
}

// List is an inner list of items.
func List(items ...Item) Item {
	parts := make([]string, len(items))
	for i, item := range items {
		parts[i] = item.String()
	}
	return Item{bare: "(" + strings.Join(parts, " ") + ")"}
}

// With adds a parameter, which is a bare item: value's own parameters are
// not kept.
func (i Item) With(key string, value Item) Item {
	if value.bare == "?1" {
		i.params += ";" + key
	} else {
		i.params += ";" + key + "=" + value.bare
	}
	return i
}

// String serializes the item with its parameters.
func (i Item) String() string { return i.bare + i.params }

// member serializes the item as the value of a dictionary member: a true
// boolean is the key alone.
func (i Item) member(key string) string {
	if i.bare == "?1" {
		return key + i.params
	}
	return key + "=" + i.String()
}
