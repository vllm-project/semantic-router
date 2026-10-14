package graph

import (
	"context"
	"errors"
	"fmt"
)

// Node is one step of a request graph. A built node is immutable and safe for
// concurrent use; every run hands it the State of the branch it runs in.
type Node interface {
	Run(ctx context.Context, x *Exec, st *State) error
}

// Step places a node in a graph under an id, which names its span and its
// evidence.
type Step struct {
	ID   string
	Type string
	Node Node
}

// Sequence runs its steps in order. It stops at the first step that fails,
// and once a step has answered the request: a respond step ends the graph
// wherever it is nested.
type Sequence []Step

// Run runs the sequence on st.
func (s Sequence) Run(ctx context.Context, x *Exec, st *State) error {
	for _, step := range s {
		if st.Response != nil {
			return nil
		}
		if err := x.runStep(ctx, step, st); err != nil {
			return err
		}
	}
	return nil
}

// Program is a built request graph, shared by every request that runs it.
type Program struct {
	// Name identifies the graph in spans and evidence, such as the Looper
	// template's name.
	Name   string
	Steps  Sequence
	Limits Limits
}

// ErrNoResponse reports a graph that ended without answering its request.
var ErrNoResponse = errors.New("graph: the program ended without a respond step")

// StepError reports the step a run failed in. Container steps pass their
// children's StepError on unchanged, so it names the innermost failing step.
type StepError struct {
	Step string
	Type string
	Err  error
}

func (e *StepError) Error() string {
	return fmt.Sprintf("graph step %q (%s): %v", e.Step, e.Type, e.Err)
}

func (e *StepError) Unwrap() error { return e.Err }

// PanicError reports a node that panicked; the run fails instead of the
// process.
type PanicError struct {
	Value any
	Stack []byte
}

func (e *PanicError) Error() string { return fmt.Sprintf("graph: node panicked: %v", e.Value) }
