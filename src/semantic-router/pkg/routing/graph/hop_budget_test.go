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
