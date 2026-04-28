package classification

import (
	"sync"
	"testing"
)

// TestRunSignalDispatchersPanicRecovery verifies that a panic inside a signal
// evaluator goroutine does not propagate to the caller. Before this fix a panic
// would crash the whole router process; now it is recovered, logged, and counted.
func TestRunSignalDispatchersPanicRecovery(t *testing.T) {
	panicDispatcher := signalDispatch{
		signalType: "keyword",
		name:       "Keyword",
		evaluate: func() {
			panic("simulated classifier out-of-bounds on long prompt")
		},
	}
	okDispatcher := signalDispatch{
		signalType: "context",
		name:       "Context",
		evaluate:   func() {},
	}

	usedSignals := map[string]bool{
		"keyword:test": true,
		"context:test": true,
	}
	ready := map[string]bool{
		"keyword": true,
		"context": true,
	}

	var wg sync.WaitGroup
	// Must not panic — the test process would crash if recovery is missing.
	runSignalDispatchers([]signalDispatch{panicDispatcher, okDispatcher}, usedSignals, ready, &wg)
	wg.Wait()
}

// TestRunSignalDispatchersMultiplePanicsRecovered ensures that a panic in one
// goroutine does not prevent other signal evaluators from completing.
func TestRunSignalDispatchersMultiplePanicsRecovered(t *testing.T) {
	var completed sync.Map

	dispatchers := []signalDispatch{
		{
			signalType: "keyword",
			name:       "Keyword",
			evaluate: func() {
				panic("keyword panic")
			},
		},
		{
			signalType: "context",
			name:       "Context",
			evaluate: func() {
				completed.Store("context", true)
			},
		},
		{
			signalType: "embedding",
			name:       "Embedding",
			evaluate: func() {
				panic("embedding panic")
			},
		},
		{
			signalType: "language",
			name:       "Language",
			evaluate: func() {
				completed.Store("language", true)
			},
		},
	}

	usedSignals := map[string]bool{
		"keyword:a":   true,
		"context:b":   true,
		"embedding:c": true,
		"language:d":  true,
	}
	ready := map[string]bool{
		"keyword":   true,
		"context":   true,
		"embedding": true,
		"language":  true,
	}

	var wg sync.WaitGroup
	runSignalDispatchers(dispatchers, usedSignals, ready, &wg)
	wg.Wait()

	for _, name := range []string{"context", "language"} {
		if _, ok := completed.Load(name); !ok {
			t.Errorf("signal %q did not complete — panic in sibling goroutine blocked it", name)
		}
	}
}
