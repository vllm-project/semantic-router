package graph

import (
	"context"
	"errors"
	"fmt"
)

// OnError is what a parallel step does when a branch fails.
type OnError string

const (
	// OnErrorFail fails the step at the first failed branch and cancels the
	// others.
	OnErrorFail OnError = "fail"
	// OnErrorSkip drops failed branches; the step fails only when fewer than
	// MinSuccess branches succeed.
	OnErrorSkip OnError = "skip"
)

// ErrTooFewResults reports a parallel step whose successful branches fell
// short of what it needs.
var ErrTooFewResults = errors.New("graph: too few branches succeeded")

// errEnough stops the branches still running once a parallel step has the
// first k successes it needs.
var errEnough = errors.New("graph: the parallel step has its first results")

// Parallel is the parallel step: it runs its branches concurrently, each on a
// fork of the state, and makes the successful branches' results the state's
// results, in branch order.
type Parallel struct {
	Branches []Sequence
	// MaxConcurrency caps the branches running at once; zero runs all.
	// Branches start in order.
	MaxConcurrency int
	// FirstK, when set, ends the step once k branches have succeeded: the
	// branches still running are cancelled and those not started never
	// start.
	FirstK int
	// MinSuccess is the number of successful branches the step needs; zero
	// means FirstK when that is set, else every branch under OnErrorFail and
	// one under OnErrorSkip.
	MinSuccess int
	OnError    OnError
}

type branchRun struct {
	results []*Result
	err     error
	kept    bool
}

// Run runs the branches. One loop starts them in order, as slots free up,
// and takes their outcomes, so a branch starts only after every outcome that
// could have stopped the step has been taken.
func (n *Parallel) Run(ctx context.Context, x *Exec, st *State) error {
	ctx, stop := context.WithCancelCause(ctx)
	defer stop(nil)
	runs := make([]branchRun, len(n.Branches))
	finished := make(chan int, len(n.Branches))
	next, running := 0, 0
	launch := func() {
		for next < len(n.Branches) && running < n.concurrency() && ctx.Err() == nil {
			i, fork := next, st.fork()
			next++
			running++
			go func() {
				runs[i].err = n.Branches[i].Run(ctx, x, fork)
				runs[i].results = fork.Results
				finished <- i
			}()
		}
	}

	kept, failed := 0, 0
	var firstErr error
	for launch(); running > 0; launch() {
		i := <-finished
		running--
		run := &runs[i]
		switch {
		case run.err == nil:
			if n.FirstK == 0 || kept < n.FirstK {
				run.kept = true
				kept++
				if kept == n.FirstK {
					stop(errEnough)
				}
			}
		case errors.Is(context.Cause(ctx), errEnough):
		default:
			failed++
			if firstErr == nil {
				firstErr = run.err
			}
			// Run-wide ceilings fail closed even when this parallel step is
			// configured to skip ordinary branch failures.
			if errors.Is(run.err, ErrHopLimit) {
				stop(run.err)
			}
			if n.OnError != OnErrorSkip {
				stop(run.err)
			}
		}
	}
	if cause := context.Cause(ctx); cause != nil && !errors.Is(cause, errEnough) &&
		(!errors.Is(cause, firstErr) || errors.Is(cause, ErrHopLimit)) {
		// The run itself was stopped: a ceiling, its timeout or the client.
		return cause
	}
	if n.OnError != OnErrorSkip && firstErr != nil {
		return firstErr
	}
	if need := n.needed(); kept < need {
		if firstErr != nil {
			return fmt.Errorf("%w: %d of %d needed (%d failed): %w", ErrTooFewResults, kept, need, failed, firstErr)
		}
		return fmt.Errorf("%w: %d of %d needed", ErrTooFewResults, kept, need)
	}
	results := make([]*Result, 0, kept)
	for i := range runs {
		if runs[i].kept {
			results = append(results, runs[i].results...)
		}
	}
	st.Results = results
	return nil
}

func (n *Parallel) concurrency() int {
	if n.MaxConcurrency > 0 && n.MaxConcurrency < len(n.Branches) {
		return n.MaxConcurrency
	}
	return max(len(n.Branches), 1)
}

func (n *Parallel) needed() int {
	switch {
	case n.MinSuccess > 0:
		return n.MinSuccess
	case n.FirstK > 0:
		return n.FirstK
	case n.OnError == OnErrorSkip:
		return 1
	default:
		return len(n.Branches)
	}
}
