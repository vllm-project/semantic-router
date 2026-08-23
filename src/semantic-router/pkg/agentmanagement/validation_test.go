package agentmanagement

import "testing"

func TestValidNameRejectsInvisibleAndControlCharacters(t *testing.T) {
	for _, value := range []string{"", " leading", "trailing ", "line\nbreak", "zero\u200bwidth", "escape\x1b"} {
		if validName(value) {
			t.Errorf("validName(%q) = true", value)
		}
	}
	for _, value := range []string{"Builder", "Recipe tuning", "多语言构建器"} {
		if !validName(value) {
			t.Errorf("validName(%q) = false", value)
		}
	}
}
