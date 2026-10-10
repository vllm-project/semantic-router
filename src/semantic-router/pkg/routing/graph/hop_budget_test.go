package graph

import (
	"errors"
	"testing"
)

// A parallel step's branches are hops of the same run, so the run's hop
// budget bounds them together and the run fails closed at the limit.
func TestParallelBranchesShareTheRunHopBudget(t *testing.T) {
	caller := &fakeCaller{}
	program := &Program{
		Steps: Sequence{
			parallel("panel", &Parallel{Branches: branchesOf("a", "b", "c", "d"), MaxConcurrency: 1}),
			aggregate("join", Concat{Separator: ","}),
			respond(),
		},
		Limits: Limits{MaxHops: 2},
	}
	outcome, err := run(t, program, caller)
	if !errors.Is(err, ErrHopLimit) {
		t.Fatalf("err %v", err)
	}
	if outcome.Hops != 2 || len(caller.requests()) != 2 {
		t.Fatalf("hops %d sent %d", outcome.Hops, len(caller.requests()))
	}
}

// Nested steps draw from the same budget: a loop running a parallel panel
// cannot amplify past the run's hop limit across rounds.
func TestNestedStepsShareTheRunHopBudget(t *testing.T) {
	caller := &fakeCaller{}
	program := &Program{
		Steps: Sequence{
			{ID: "rounds", Type: TypeLoop, Node: &Loop{
				Body:      Sequence{parallel("panel", &Parallel{Branches: branchesOf("a", "b"), MaxConcurrency: 1})},
				MaxRounds: 5,
			}},
			respond(),
		},
		Limits: Limits{MaxHops: 3},
	}
	outcome, err := run(t, program, caller)
	if !errors.Is(err, ErrHopLimit) {
		t.Fatalf("err %v", err)
	}
	if outcome.Hops != 3 {
		t.Fatalf("hops %d", outcome.Hops)
	}
}

// A parallel step may skip ordinary branch failures, but exhausting the
// request-wide hop budget must still fail the whole run.
func TestParallelSkipCannotHideHopBudgetExhaustion(t *testing.T) {
	program := &Program{
		Steps: Sequence{
			parallel("panel", &Parallel{Branches: branchesOf("a", "b"), MaxConcurrency: 1, OnError: OnErrorSkip}),
			aggregate("join", Concat{Separator: ","}),
			respond(),
		},
		Limits: Limits{MaxHops: 1},
	}
	outcome, err := run(t, program, &fakeCaller{})
	if !errors.Is(err, ErrHopLimit) {
		t.Fatalf("err %v, want ErrHopLimit", err)
	}
	if outcome.Hops != 1 {
		t.Fatalf("hops %d, want 1", outcome.Hops)
	}
}
