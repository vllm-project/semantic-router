package extproc

import (
	"fmt"
	"strings"
)

const maximumDurableResourceIDBytes = 128

// durableResourceID preserves the immutable identifier published by the
// routing control plane. Durable routing resources use bounded canonical
// codes (for example ep_..., rcp_..., dec_..., and mdl_...), not UUIDs.
func durableResourceID(label, value string) (string, error) {
	if value == "" || len(value) > maximumDurableResourceIDBytes || strings.TrimSpace(value) != value {
		return "", fmt.Errorf("%s is not a bounded canonical resource ID", label)
	}
	for index, character := range []byte(value) {
		if character >= 'A' && character <= 'Z' || character >= 'a' && character <= 'z' ||
			character >= '0' && character <= '9' || (index > 0 && strings.ContainsRune("._:/-", rune(character))) {
			continue
		}
		return "", fmt.Errorf("%s is not a bounded canonical resource ID", label)
	}
	return value, nil
}
