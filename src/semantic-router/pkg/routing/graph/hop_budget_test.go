package graph

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
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

func TestNestedParallelSkipCannotHideRunCeilings(t *testing.T) {
	tests := []struct {
		name    string
		limits  Limits
		pricing Pricing
		want    error
		calls   int
	}{
		{name: "tokens", limits: Limits{MaxTokens: 1}, want: ErrTokenLimit, calls: 1},
		{name: "cost", limits: Limits{MaxCost: 1}, pricing: pricePerToken{"a": 1}, want: ErrCostLimit, calls: 1},
		{name: "unknown cost", limits: Limits{MaxCost: 2}, pricing: pricePerToken{"a": 1}, want: ErrCostUnknown, calls: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			caller := &fakeCaller{}
			program := &Program{
				Steps: Sequence{
					parallel("outer", &Parallel{
						Branches: []Sequence{
							{call("first", "a")},
							{parallel("inner", &Parallel{Branches: branchesOf("b")})},
						},
						MaxConcurrency: 1,
						MinSuccess:     1,
						OnError:        OnErrorSkip,
					}),
					aggregate("join", Concat{Separator: ","}),
					respond(),
				},
				Limits: tt.limits,
			}
			outcome, err := run(t, program, caller, func(o *Options) { o.Pricing = tt.pricing })
			if !errors.Is(err, tt.want) {
				t.Fatalf("err %v, want %v", err, tt.want)
			}
			if outcome.Response != nil || len(caller.requests()) != tt.calls {
				t.Fatalf("response %+v, sent %d calls, want %d", outcome.Response, len(caller.requests()), tt.calls)
			}
		})
	}
}

func TestPhysicalRetryBudgetExhaustionFailsTheGraph(t *testing.T) {
	calls := 0
	program := &Program{
		Steps:  Sequence{call("retry", "model"), respond()},
		Limits: Limits{MaxHops: 1},
	}
	outcome, err := run(t, program, &budgetCaller{calls: &calls})
	if !errors.Is(err, ErrHopLimit) {
		t.Fatalf("err %v, want ErrHopLimit", err)
	}
	if calls != 1 || outcome.Hops != 1 || outcome.PhysicalCalls != 1 {
		t.Fatalf("physical calls=%d ledger=%d logical hops=%d, want one of each", calls, outcome.PhysicalCalls, outcome.Hops)
	}
}

type budgetCaller struct{ calls *int }

func (c *budgetCaller) Call(ctx context.Context, req *HopRequest) (*HopResponse, error) {
	if err := budget.Consume(ctx); err != nil {
		return nil, err
	}
	*c.calls++
	// Model one upstream attempt followed by a retry inside the same logical
	// hop. The graph must account for the retry even though Caller.Call is
	// entered only once.
	if err := budget.Consume(ctx); err != nil {
		return nil, err
	}
	return completion(req.Model, "ok", 1), nil
}
