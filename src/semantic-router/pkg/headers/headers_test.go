package headers

import (
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
