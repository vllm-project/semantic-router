// Package extproc — small helper for goroutine panic recovery.
//
// Several extproc background goroutines (config-reload debouncer, async
// memory store, parallel AR text + diffusion calls, lookup-table
// populator) historically had no panic recovery. A panic inside any of
// them propagated up and crashed the whole router process with no log
// line — the symptom reported in
// https://github.com/vllm-project/semantic-router/issues/1843, where a
// ~4000-token prompt silently took the container down.
//
// `goSafely(name, fn)` wraps `go fn()` with a deferred recover that
// logs the panic + a short stack snippet via the existing structured
// logger and returns. The caller's normal error-handling path is
// unchanged for non-panic outcomes.

package extproc

import (
	"runtime/debug"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// goSafely launches fn in a new goroutine with a deferred panic
// recovery that emits a structured `extproc:goroutine_panic` event
// instead of letting the panic abort the process. `name` identifies
// the call site in the resulting log entry.
func goSafely(name string, fn func()) {
	go func() {
		defer func() {
			if r := recover(); r != nil {
				logging.ComponentErrorEvent("extproc", "goroutine_panic", map[string]interface{}{
					"goroutine": name,
					"panic":     r,
					"stack":     string(debug.Stack()),
				})
			}
		}()
		fn()
	}()
}
