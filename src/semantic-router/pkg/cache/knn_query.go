package cache

import (
	"fmt"
	"strings"
)

// escapeTagValue escapes punctuation and whitespace in a string so it can be
// safely used inside a Valkey/Redis TAG query expression (@field:{value}).
// TAG queries treat punctuation characters as token separators; backslash-
// escaping them preserves the literal value.
// Reference: https://forum.redis.com/t/tag-fields-and-escaping/96
func escapeTagValue(s string) string {
	// Characters that must be backslash-escaped in TAG query values.
	// These are the punctuation/space characters that the RediSearch/Valkey
	// query tokenizer treats as separators.
	const specialChars = " \t,.<>{}[]\"':;!@#$%^&*()-+=~|/\\"

	var b strings.Builder
	b.Grow(len(s) + 8) // most IDs need only a few extra backslashes
	for _, c := range s {
		if strings.ContainsRune(specialChars, c) {
			b.WriteByte('\\')
		}
		b.WriteRune(c)
	}
	return b.String()
}

// partitionedKNNQuery combines an exact model TAG filter with vector search.
// The model field is the cache partition key, so omitting this filter can
// return another model or recipe's response even when its vector is nearest.
func partitionedKNNQuery(model string, topK int, vectorField string) string {
	return fmt.Sprintf(
		"(@model:{%s})=>[KNN %d @%s $vec AS vector_distance]",
		escapeTagValue(model),
		topK,
		vectorField,
	)
}
