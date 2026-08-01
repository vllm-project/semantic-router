package candle_binding

// This file carries no build constraint on purpose. Request validation must run
// identically whether the native Candle backend is linked (CGO build) or the
// fail-closed stub is compiled in (non-CGO build). Both semantic-router.go and
// semantic-router_mock.go call these helpers before dispatching a request, so a
// malformed input is rejected the same way in either mode rather than only when
// the native backend happens to be present (issue #2619).
//
// Scope: this is the safe subset — presence and NUL-byte checks only. Semantic
// rules (target-dimension legal sets, top-k bounds, tensor-shape constraints)
// are tracked as follow-up in #2675.
//
// The NUL check does tighten one previously accepted case: a data URI whose
// prefix contained a NUL, such as "data:image/png\x00;base64,...", used to
// decode successfully because everything before ";base64," is discarded. That
// input is now rejected. See validateRequiredText for why that is intended.

import (
	"fmt"
	"strings"
)

// validateRequiredText validates a required string argument to a public
// multimodal entry point. It rejects empty values and values containing a NUL
// byte. The field name is interpolated verbatim so the message matches the
// historical inline checks, e.g. "text cannot be empty".
//
// The reason for rejecting NUL differs by argument, and only one of them is a
// cgo concern:
//
//   - text is passed to C.CString, where a NUL terminates the string early, so
//     the native backend would silently receive a truncated prompt.
//   - base64Str and url never reach cgo: they are consumed in Go by
//     base64.StdEncoding.DecodeString and http.NewRequest respectively. A NUL
//     there is not a truncation risk but a malformed-input signal — in a URL it
//     is a classic request-smuggling smell, and in a data URI it can hide bytes
//     ahead of the ";base64," separator that the parser then discards.
//
// Rejecting NUL uniformly is deliberate: a shared validator that applied
// different rules per argument would be harder to reason about than one that
// refuses a byte no legitimate caller sends. Callers that genuinely need to
// carry NUL in a payload should encode it rather than pass it raw.
func validateRequiredText(field, value string) error {
	if value == "" {
		return fmt.Errorf("%s cannot be empty", field)
	}
	if strings.IndexByte(value, 0) >= 0 {
		return fmt.Errorf("%s cannot contain NUL bytes", field)
	}
	return nil
}
