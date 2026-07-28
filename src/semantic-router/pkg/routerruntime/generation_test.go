package routerruntime

import (
	"errors"
	"fmt"
	"testing"
)

func TestGenerationCloseRunsClosersInReverseOrder(t *testing.T) {
	var order []int
	gen := NewGeneration()
	for i := 0; i < 3; i++ {
		i := i
		gen.Defer(func() error {
			order = append(order, i)
			return nil
		})
	}

	if err := gen.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	want := []int{2, 1, 0}
	if len(order) != len(want) {
		t.Fatalf("order = %v, want %v", order, want)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("order = %v, want %v", order, want)
		}
	}
}

func TestGenerationCloseRunsClosersOnlyOnce(t *testing.T) {
	calls := 0
	gen := NewGeneration()
	gen.Defer(func() error {
		calls++
		return nil
	})

	if err := gen.Close(); err != nil {
		t.Fatalf("first Close() error = %v", err)
	}
	if err := gen.Close(); err != nil {
		t.Fatalf("second Close() error = %v", err)
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1", calls)
	}
}

func TestGenerationCloseJoinsErrorsFromEveryCloser(t *testing.T) {
	errA := errors.New("a")
	errB := errors.New("b")
	gen := NewGeneration()
	gen.Defer(func() error { return errA })
	gen.Defer(func() error { return errB })

	err := gen.Close()
	if !errors.Is(err, errA) || !errors.Is(err, errB) {
		t.Fatalf("Close() error = %v, want it to wrap both %v and %v", err, errA, errB)
	}
}

func TestGenerationDeferIgnoresNilClosers(t *testing.T) {
	gen := NewGeneration()
	gen.Defer(nil)

	if err := gen.Close(); err != nil {
		t.Fatalf("Close() error = %v, want nil", err)
	}
}

// TestGenerationFaultInjectionExactOnceReverseCleanupAtEachStep simulates a
// constructor with several steps, each registering a closer immediately
// after succeeding (buildRouterComponents's pattern), and injects a failure
// at every possible step in turn. For each injected position, it asserts
// that exactly the closers registered before the failure run — each exactly
// once, in reverse order — regardless of where in the sequence construction
// failed. This is the issue's own fault-injection validation ask.
func TestGenerationFaultInjectionExactOnceReverseCleanupAtEachStep(t *testing.T) {
	const steps = 6
	for failAt := 0; failAt < steps; failAt++ {
		t.Run(fmt.Sprintf("fail_at_step_%d", failAt), func(t *testing.T) {
			var closeOrder []int
			var closeCounts [steps]int
			gen := NewGeneration()
			var built int
			for step := 0; step < steps; step++ {
				if step == failAt {
					break
				}
				step := step
				gen.Defer(func() error {
					closeOrder = append(closeOrder, step)
					closeCounts[step]++
					return nil
				})
				built++
			}
			if built != failAt {
				t.Fatalf("test setup error: registered %d closers, want %d before failing at step %d", built, failAt, failAt)
			}

			// The failed constructor step itself rolls back via gen.Close(),
			// mirroring rollbackGeneration in router_build.go.
			if err := gen.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}

			if len(closeOrder) != failAt {
				t.Fatalf("closed %d resources, want %d (steps registered before the injected failure)", len(closeOrder), failAt)
			}
			for step, count := range closeCounts {
				switch {
				case step < failAt && count != 1:
					t.Fatalf("step %d closer ran %d times, want exactly 1", step, count)
				case step >= failAt && count != 0:
					t.Fatalf("step %d closer ran %d times, want 0 (never registered — construction failed before this step)", step, count)
				}
			}
			for i := 1; i < len(closeOrder); i++ {
				if closeOrder[i] >= closeOrder[i-1] {
					t.Fatalf("closers did not run in reverse registration order: %v", closeOrder)
				}
			}

			// A caller may still hold the generation after a failed build
			// (e.g. a deferred cleanup); Close must stay idempotent.
			if err := gen.Close(); err != nil {
				t.Fatalf("second Close() error = %v", err)
			}
			if len(closeOrder) != failAt {
				t.Fatalf("second Close() re-ran closers: got %d calls, want %d", len(closeOrder), failAt)
			}
		})
	}
}

func TestNilGenerationCloseIsNoop(t *testing.T) {
	var gen *Generation
	if err := gen.Close(); err != nil {
		t.Fatalf("Close() error = %v, want nil", err)
	}
	gen.Defer(func() error { return errors.New("should never run") })
}
