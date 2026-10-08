package graph

import (
	"context"
	"errors"
	"fmt"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// Limits bound one run of a program; a zero field means no bound. A run that
// exhausts one fails closed: the hop that would cross the hop limit is not
// sent, a hop whose usage crosses a ceiling fails the run, and failing the
// run cancels every hop still in flight.
type Limits struct {
	// Timeout bounds the run, within the request's own deadline.
	Timeout   time.Duration
	MaxHops   int
	MaxTokens int64
	// MaxCost is in the currency Options.Pricing prices in.
	MaxCost float64
}

// Errors a run fails closed with.
var (
	ErrDeadline    = errors.New("graph: the run's timeout passed")
	ErrHopLimit    = errors.New("graph: the hop limit is reached")
	ErrTokenLimit  = errors.New("graph: the token ceiling is reached")
	ErrCostLimit   = errors.New("graph: the cost ceiling is reached")
	ErrCostUnknown = errors.New("graph: a cost ceiling needs the price of every model it calls")
)

// Pricing prices a call's usage for a cost ceiling.
type Pricing interface {
	Cost(model string, usage Usage) (cost float64, known bool)
}

// Options configure one run.
type Options struct {
	// Caller sends the run's hops.
	Caller Caller
	// Hop is the request's routing context: the decision whose plugin chain
	// a call step's hop runs unless the step names another, and its recipe.
	Hop routing.Hop
	// Header is what a call step's hop request header starts from,
	// pseudo-headers included; nil means ChatHeader().
	Header routing.Header
	// Pricing prices hops when the program has a cost ceiling.
	Pricing Pricing
	// Signals are the request's matched signals by type, for conditions.
	Signals map[string][]string
}

// Input is what a run starts from: the client's chat request after its
// request-side plugins, and typed values for the steps that need them.
type Input struct {
	Request Request
	Values  map[string]any
}

// Outcome is what a run produced, with its evidence. A failed run still
// reports the hops it sent.
type Outcome struct {
	Response *Response
	// Results are the last step's results.
	Results []*Result
	Values  map[string]any
	// Hops counts the hops sent; Usage and Cost add up their usage.
	Hops            int
	Usage           Usage
	Cost            float64
	Attempts        []Attempt
	DroppedAttempts int
}

// Exec is one run of a program. Its methods are safe for concurrent use by
// the branches of a parallel step.
type Exec struct {
	program *Program
	opts    Options
	fail    context.CancelCauseFunc

	hops     atomic.Int64
	mu       sync.Mutex
	usage    Usage
	cost     float64
	evidence evidence
}

// Run executes program for one request. ctx carries the request's deadline
// and cancellation; a client that goes away cancels every hop in flight.
func Run(ctx context.Context, program *Program, in Input, opts Options) (*Outcome, error) {
	if program == nil {
		return nil, errors.New("graph: a program is required")
	}
	if opts.Caller == nil {
		return nil, errors.New("graph: a caller is required")
	}
	if opts.Header == nil {
		opts.Header = ChatHeader()
	}
	runCtx, fail := context.WithCancelCause(ctx)
	defer fail(nil)
	if program.Limits.Timeout > 0 {
		var stop context.CancelFunc
		runCtx, stop = context.WithTimeoutCause(runCtx, program.Limits.Timeout, ErrDeadline)
		defer stop()
	}
	x := &Exec{program: program, opts: opts, fail: fail}
	st := &State{Request: in.Request.Clone(), Values: cloneValues(in.Values)}

	spanCtx, span := tracing.StartSpan(runCtx, "graph.run",
		trace.WithAttributes(attribute.String(attrProgram, program.Name)))
	err := program.Steps.Run(spanCtx, x, st)
	if err == nil && st.Response == nil {
		err = ErrNoResponse
	}
	if err != nil {
		err = runError(runCtx, err)
		tracing.RecordError(span, errorReason(err))
	}
	outcome := x.outcome(st)
	span.SetAttributes(attribute.Int(attrHops, outcome.Hops))
	span.End()
	if err != nil {
		return outcome, err
	}
	return outcome, nil
}

// runError prefers why the run was stopped (a ceiling, the timeout, the
// client) over the error a step saw because of it.
func runError(runCtx context.Context, err error) error {
	cause := context.Cause(runCtx)
	if cause == nil || errors.Is(err, cause) {
		return err
	}
	return fmt.Errorf("%w (%w)", cause, err)
}

func cloneValues(values map[string]any) map[string]any {
	out := make(map[string]any, len(values))
	for name, value := range values {
		out[name] = value
	}
	return out
}

// Hop returns the request's routing context.
func (x *Exec) Hop() routing.Hop { return x.opts.Hop }

// Signals returns the request's matched signals of one type.
func (x *Exec) Signals(signalType string) []string { return x.opts.Signals[signalType] }

// Header returns a copy of the header call steps start their hop requests
// from.
func (x *Exec) Header() routing.Header { return x.opts.Header.Clone() }

// Fail stops the run with cause; every hop in flight is cancelled.
func (x *Exec) Fail(cause error) { x.fail(cause) }

func (x *Exec) outcome(st *State) *Outcome {
	x.mu.Lock()
	defer x.mu.Unlock()
	attempts, dropped := x.evidence.snapshot()
	return &Outcome{
		Response:        st.Response,
		Results:         st.Results,
		Values:          st.Values,
		Hops:            int(x.hops.Load()),
		Usage:           x.usage,
		Cost:            x.cost,
		Attempts:        attempts,
		DroppedAttempts: dropped,
	}
}

type stepKey struct{}

// runStep runs one step under its span and its duration metric. A panic fails
// the step, not the process; an error is attributed to the innermost step it
// came from. A step starts even when the run is already stopped, so it
// observes the stop the way it observes any failed hop: its hops are refused
// at admission.
func (x *Exec) runStep(ctx context.Context, step Step, st *State) (err error) {
	started := time.Now()
	ctx, span := tracing.StartSpan(ctx, "graph."+step.Type, trace.WithAttributes(
		attribute.String(attrProgram, x.program.Name),
		attribute.String(attrStepID, step.ID),
		attribute.String(attrStepType, step.Type),
	))
	ctx = context.WithValue(ctx, stepKey{}, step.ID)
	defer func() {
		if recovered := recover(); recovered != nil {
			err = &PanicError{Value: recovered, Stack: debug.Stack()}
		}
		var stepErr *StepError
		if err != nil && !errors.As(err, &stepErr) && context.Cause(ctx) == nil {
			err = &StepError{Step: step.ID, Type: step.Type, Err: err}
		}
		if err != nil {
			tracing.RecordError(span, errorReason(err))
		}
		span.End()
		metrics.RecordRequestGraphNode(step.Type, x.program.Name, time.Since(started).Seconds())
	}()
	return step.Node.Run(ctx, x, st)
}

// stepOf returns the id of the step ctx runs in.
func stepOf(ctx context.Context) string {
	id, _ := ctx.Value(stepKey{}).(string)
	return id
}

// errorReason is a bounded, content-free reason for a span.
func errorReason(err error) string {
	var callErr *CallError
	var panicErr *PanicError
	switch {
	case errors.Is(err, ErrHopLimit):
		return "hop_limit"
	case errors.Is(err, ErrTokenLimit):
		return "token_ceiling"
	case errors.Is(err, ErrCostLimit), errors.Is(err, ErrCostUnknown):
		return "cost_ceiling"
	case errors.Is(err, ErrDeadline), errors.Is(err, context.DeadlineExceeded):
		return "deadline"
	case errors.Is(err, context.Canceled):
		return "cancelled"
	case errors.As(err, &panicErr):
		return "panic"
	case errors.As(err, &callErr):
		return "call_failed"
	case errors.Is(err, ErrNoResponse):
		return "no_response"
	default:
		return "step_failed"
	}
}

// Span attribute keys.
const (
	attrProgram     = "graph.program"
	attrStepID      = "graph.step.id"
	attrStepType    = "graph.step.type"
	attrHops        = "graph.hops"
	attrHopModel    = "graph.hop.model"
	attrHopOrdinal  = "graph.hop.ordinal"
	attrHopStatus   = "graph.hop.status"
	attrBranchTaken = "graph.branch.case"
	attrLoopRounds  = "graph.loop.rounds"
)
