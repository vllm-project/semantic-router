package graph

import (
	"context"
	"errors"
	"fmt"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// HopRequest is one model call of a run.
type HopRequest struct {
	// Hop is the call's routing context; a zero Iteration takes the hop's
	// ordinal in the run.
	Hop routing.Hop
	// Model names the model called, for budgets and evidence.
	Model string
	// Request is the call as its session receives it, pseudo-headers first.
	Request routing.Request
	// Reliability layers over the decision's reliability for this call.
	Reliability *routing.Reliability
}

// HopResponse is a hop's answer as the client side of its session sees it.
type HopResponse struct {
	Status int
	Header routing.Header
	Body   []byte
}

// Caller sends hops. SessionCaller is the routing core's in-process caller.
type Caller interface {
	Call(ctx context.Context, req *HopRequest) (*HopResponse, error)
}

// ChatHeader is the header of a chat completions call.
func ChatHeader() routing.Header {
	return routing.Header{
		{Name: ":method", Value: "POST"},
		{Name: ":path", Value: "/v1/chat/completions"},
		{Name: ":scheme", Value: "http"},
		{Name: ":authority", Value: "localhost"},
		{Name: "content-type", Value: "application/json"},
	}
}

// Call sends one hop under the run's budget: it is refused when the run is
// over a limit, its usage is charged against the ceilings once it returns,
// and its span and attempt evidence are recorded. Every model call of a run,
// a call step's or a strategy's, goes through it.
func (x *Exec) Call(ctx context.Context, req *HopRequest) (*HopResponse, error) {
	ordinal, err := x.admit(ctx)
	if err != nil {
		return nil, err
	}
	sent := *req
	if sent.Hop.Iteration == 0 {
		sent.Hop.Iteration = ordinal
	}

	ctx, span := tracing.StartSpan(ctx, "graph.hop", trace.WithAttributes(
		attribute.String(attrProgram, x.program.Name),
		attribute.String(attrStepID, stepOf(ctx)),
		attribute.String(attrHopModel, req.Model),
		attribute.Int(attrHopOrdinal, ordinal),
	))
	defer span.End()
	started := time.Now()
	resp, err := x.opts.Caller.Call(ctx, &sent)
	latency := time.Since(started)

	var usage Usage
	status := 0
	if resp != nil {
		usage = completionUsage(resp.Body)
		status = resp.Status
		span.SetAttributes(attribute.Int(attrHopStatus, status))
	}
	chargeErr := x.charge(req.Model, usage)
	x.record(ctx, ordinal, req.Model, status, usage, latency, errors.Join(err, chargeErr))
	switch {
	case err != nil:
		tracing.RecordError(span, errorReason(err))
		return nil, err
	case chargeErr != nil:
		tracing.RecordError(span, errorReason(chargeErr))
		return nil, chargeErr
	}
	return resp, nil
}

// admit refuses a hop the run cannot send and numbers the others.
func (x *Exec) admit(ctx context.Context) (int, error) {
	if cause := context.Cause(ctx); cause != nil {
		return 0, cause
	}
	limits := x.program.Limits
	ordinal := int(x.hops.Add(1))
	if limits.MaxHops > 0 && ordinal > limits.MaxHops {
		x.hops.Add(-1)
		x.fail(ErrHopLimit)
		return 0, ErrHopLimit
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	switch {
	case limits.MaxTokens > 0 && x.usage.TotalTokens >= limits.MaxTokens:
		x.hops.Add(-1)
		x.fail(ErrTokenLimit)
		return 0, ErrTokenLimit
	case limits.MaxCost > 0 && x.cost >= limits.MaxCost:
		x.hops.Add(-1)
		x.fail(ErrCostLimit)
		return 0, ErrCostLimit
	}
	return ordinal, nil
}

// charge adds a hop's usage to the run and fails the run once a ceiling is
// crossed.
func (x *Exec) charge(model string, usage Usage) error {
	limits := x.program.Limits
	x.mu.Lock()
	defer x.mu.Unlock()
	x.usage = x.usage.Add(usage)
	if limits.MaxCost > 0 && usage != (Usage{}) {
		price, known := 0.0, false
		if x.opts.Pricing != nil {
			price, known = x.opts.Pricing.Cost(model, usage)
		}
		if !known {
			x.fail(ErrCostUnknown)
			return fmt.Errorf("%w (model %q)", ErrCostUnknown, model)
		}
		x.cost += price
	}
	switch {
	case limits.MaxTokens > 0 && x.usage.TotalTokens > limits.MaxTokens:
		x.fail(ErrTokenLimit)
		return ErrTokenLimit
	case limits.MaxCost > 0 && x.cost > limits.MaxCost:
		x.fail(ErrCostLimit)
		return ErrCostLimit
	}
	return nil
}
