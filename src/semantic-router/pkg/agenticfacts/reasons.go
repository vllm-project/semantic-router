package agenticfacts

// Rejection reason codes. These are a stable contract: they appear in Replay
// provenance and operator diagnostics, so existing codes keep their spelling and
// meaning across schema versions even as fields come and go.
//
// The codes name what was wrong, never what was sent. Diagnostics stay
// content-minimized: field names, reason codes, and counts, never field values.
const (
	ReasonMissing            = "missing"
	ReasonMalformed          = "malformed"
	ReasonUnsupportedVersion = "unsupported_version"
	ReasonExpired            = "expired"
	ReasonTooLarge           = "too_large"
	ReasonTooDeep            = "too_deep"
	ReasonTooMany            = "too_many"
	ReasonTooLong            = "too_long"
	ReasonConflicting        = "conflicting"
	ReasonUntrusted          = "untrusted"
)

var validReasons = []string{
	ReasonMissing,
	ReasonMalformed,
	ReasonUnsupportedVersion,
	ReasonExpired,
	ReasonTooLarge,
	ReasonTooDeep,
	ReasonTooMany,
	ReasonTooLong,
	ReasonConflicting,
	ReasonUntrusted,
}

// IsKnownReason reports whether a reason code is one this package emits. It
// guards consumers that read reason codes back out of Replay records written by
// a different build.
func IsKnownReason(r string) bool {
	for _, reason := range validReasons {
		if r == reason {
			return true
		}
	}
	return false
}
