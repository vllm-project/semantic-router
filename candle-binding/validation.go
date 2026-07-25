package candle_binding

// This file carries no build constraint on purpose. Request validation must run
// identically whether the native Candle backend is linked (CGO build) or the
// fail-closed stub is compiled in (non-CGO build). Both semantic-router.go and
// semantic-router_mock.go call these helpers before dispatching a request, so a
// malformed input is rejected the same way in either mode rather than only when
// the native backend happens to be present (issue #2619).
//
// Scope: this is the safe subset — presence and NUL-byte checks that change no
// previously accepted input. Semantic rules (target-dimension legal sets,
// top-k bounds, tensor-shape constraints) are tracked as follow-up in #2619.

import (
	"fmt"
	"strings"
)

// validateRequiredText validates a required string argument that is passed
// through cgo to the native backend. It rejects empty values and values that
// contain a NUL byte, which cannot cross the cgo boundary intact (C.CString
// would truncate or error). The field name is interpolated verbatim so the
// message matches the historical inline checks, e.g. "text cannot be empty".
func validateRequiredText(field, value string) error {
	if value == "" {
		return fmt.Errorf("%s cannot be empty", field)
	}
	if strings.IndexByte(value, 0) >= 0 {
		return fmt.Errorf("%s cannot contain NUL bytes", field)
	}
	return nil
}
