package headers

import (
	"slices"
	"testing"
)

func TestWarningCodes(t *testing.T) {
	// Assert the constants directly — a map keyed by the constant would
	// compare each value to itself and never fail.
	for _, tt := range []struct{ got, want string }{
		{WarningHallucination, "hallucination"},
		{WarningUnverifiedFactual, "unverified_factual"},
		{WarningJailbreak, "response_jailbreak"},
		{WarningGuard, "response_guard"},
		{WarningProtocol, "protocol"},
		{WarningTruncated, "truncated"},
	} {
		if tt.got != tt.want {
			t.Errorf("warning code = %q, want %q", tt.got, tt.want)
		}
	}
}

func TestWarningCodesDocumentEachCode(t *testing.T) {
	want := []string{WarningHallucination, WarningUnverifiedFactual, WarningJailbreak, WarningGuard, WarningProtocol, WarningTruncated}
	var got []string
	for _, code := range WarningCodes() {
		got = append(got, code.Code)
		if !tableSafeSentence(code.Doc) {
			t.Errorf("%s: doc %q must be one sentence that a Markdown table and MDX can hold", code.Code, code.Doc)
		}
	}
	if !slices.Equal(got, want) {
		t.Errorf("WarningCodes() = %v, want %v", got, want)
	}
}
