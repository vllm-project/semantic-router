package graph

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// Case is one alternative of a branch step.
type Case struct {
	When Condition
	Then Sequence
}

// Branch is the branch step: it runs the first case whose condition holds,
// or Else when none does.
type Branch struct {
	Cases []Case
	Else  Sequence
}

// Run picks and runs a case.
func (n *Branch) Run(ctx context.Context, x *Exec, st *State) error {
	span := trace.SpanFromContext(ctx)
	for i, c := range n.Cases {
		holds, err := c.When.Holds(ctx, x, st)
		if err != nil {
			return err
		}
		if holds {
			span.SetAttributes(attribute.Int(attrBranchTaken, i))
			return c.Then.Run(ctx, x, st)
		}
	}
	span.SetAttributes(attribute.Int(attrBranchTaken, -1))
	return n.Else.Run(ctx, x, st)
}

// Loop is the loop step: it runs Body round after round on the same state
// until Until holds after a round, a round answers the request, or MaxRounds
// rounds have run. Round holds the current round.
type Loop struct {
	Body      Sequence
	Until     Condition
	MaxRounds int
}

// Run runs the rounds.
func (n *Loop) Run(ctx context.Context, x *Exec, st *State) error {
	span := trace.SpanFromContext(ctx)
	for round := 1; round <= n.MaxRounds; round++ {
		Round.Set(st, round)
		span.SetAttributes(attribute.Int(attrLoopRounds, round))
		if err := n.Body.Run(ctx, x, st); err != nil {
			return err
		}
		if st.Response != nil || n.Until == nil {
			continue
		}
		done, err := n.Until.Holds(ctx, x, st)
		if err != nil || done {
			return err
		}
	}
	return nil
}

// Subgraph runs a named, reusable sequence in place.
type Subgraph struct {
	Name  string
	Steps Sequence
}

// Run runs the subgraph's steps.
func (n *Subgraph) Run(ctx context.Context, x *Exec, st *State) error {
	return n.Steps.Run(ctx, x, st)
}

// Aggregator combines the latest results, usually into one.
type Aggregator interface {
	Aggregate(ctx context.Context, x *Exec, st *State) error
}

// Aggregate is the aggregate step.
type Aggregate struct {
	Strategy Aggregator
}

// Run applies the strategy.
func (n *Aggregate) Run(ctx context.Context, x *Exec, st *State) error {
	return n.Strategy.Aggregate(ctx, x, st)
}

// Transformer rewrites the state, usually the request the next call sends.
type Transformer interface {
	Transform(ctx context.Context, x *Exec, st *State) error
}

// Transform is the transform step.
type Transform struct {
	Transformer Transformer
}

// Run applies the transformer.
func (n *Transform) Run(ctx context.Context, x *Exec, st *State) error {
	return n.Transformer.Transform(ctx, x, st)
}

// ErrNoResult reports a respond step without exactly one successful result
// to answer with.
var ErrNoResult = errors.New("graph: respond needs exactly one successful result; aggregate first")

// Respond is the respond step: it answers the request with the latest
// result, or hands the final model call to the gateway. The final call is
// an ordinary routed call with its decision's reliability; a step's
// override applies only to the hops the run sends itself.
type Respond struct {
	// Final, when set, names the model the gateway sends the state's request
	// to as the final call, so that model's answer streams to the client.
	Final string
}

// Run answers the request.
func (n *Respond) Run(ctx context.Context, x *Exec, st *State) error {
	if n.Final != "" {
		st.Response = &Response{Final: &Final{Model: n.Final, Request: st.Request.Clone()}}
		return nil
	}
	if len(st.Results) != 1 || !st.Results[0].OK() {
		return fmt.Errorf("%w (%d results)", ErrNoResult, len(st.Results))
	}
	result := st.Results[0]
	header := routing.Header{{Name: "content-type", Value: contentType(result)}}
	st.Response = &Response{Answer: &routing.Response{Status: http.StatusOK, Header: header, Body: result.Body}}
	return nil
}

func contentType(result *Result) string {
	if value := result.Header.Get("content-type"); value != "" {
		return value
	}
	return "application/json"
}
