package candle_binding

// This file carries no build constraint on purpose: the backend-unavailable
// sentinel must be referenceable from every build so consumers can assert on it
// with errors.Is regardless of whether they were compiled with or without CGO.
// Only the non-CGO stub (semantic-router_mock.go) actually returns it.

import "errors"

// ErrBackendUnavailable is returned by every inference and mutation API in the
// non-CGO build to signal that the native Candle backend is not linked. It is a
// typed sentinel so callers can detect the unavailable-backend condition with
// errors.Is and fail closed instead of treating the result as a valid verdict.
var ErrBackendUnavailable = errors.New("candle: native backend unavailable (built without cgo)")
