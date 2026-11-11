package systemonecodec

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// Send makes one wire call: it posts body (call's path and headers) and
// returns the reply, an error status's among them; err is set when there is
// no reply.
type Send func(ctx context.Context, call WireCall, body []byte) (WireReply, error)

// Observer hears what an exchange does, for metrics and logs; every field is
// optional.
type Observer struct {
	// Diagnostics are what a codec or deviation changed in a call.
	Diagnostics func(Diagnostics)
	// Retried is a call asked again without a question the server named as
	// failed.
	Retried func(question, questionError string)
	// Replied is a call's outcome: its reply, or the error without one.
	Replied func(reply WireReply, err error, elapsed time.Duration)
}

// Exchange asks a target in one schema, through its deviations.
type Exchange struct {
	Codec      SchemaCodec
	Deviations []Deviation
	Target     Target
	Policy     Policy
	// MaxConcurrency bounds the calls in flight when items go one per call:
	// a sliding window, each answer starting the next.
	MaxConcurrency int
	Send           Send
	Observer       Observer
}

// Ask asks every item of a batch. A schema with batches gets the batch in
// one call; when that call fails for one named question, or the schema has
// none, each item goes in a call of its own, at most MaxConcurrency in
// flight. An item's call is made once, once more without a question the
// server names as failed (which then answers with its error), and once more
// after a 429's or 529's Retry-After when it fits the deadline. planErrs,
// when set, fail an item before it is sent.
func (e Exchange) Ask(ctx context.Context, batch systemoneprotocol.Batch, planErrs []error) []ItemResult {
	results := make([]ItemResult, len(batch.Items))
	if planErrs == nil {
		planErrs = make([]error, len(batch.Items))
	}
	pending := make([]int, 0, len(batch.Items))
	for index := range batch.Items {
		if planErrs[index] != nil {
			results[index].Err = &systemoneprotocol.Problem{Status: http.StatusBadRequest, Kind: systemoneprotocol.ProblemRejected, Detail: planErrs[index].Error()}
			continue
		}
		pending = append(pending, index)
	}
	if len(pending) > 1 && e.Codec.Capabilities().Has(CapabilityBatch) {
		whole := batch
		whole.Items = make([]systemoneprotocol.NamedRequest, len(pending))
		for position, index := range pending {
			whole.Items[position] = batch.Items[index]
		}
		answered, retry := e.callBatch(ctx, whole)
		if !retry {
			for position, index := range pending {
				results[index] = answered[position]
			}
			return results
		}
	}
	window := make(chan struct{}, max(1, e.MaxConcurrency))
	var wg sync.WaitGroup
	for _, index := range pending {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			select {
			case window <- struct{}{}:
			case <-ctx.Done():
				results[index].Err = ctx.Err()
				return
			}
			defer func() { <-window }()
			results[index] = e.askItem(ctx, batch, batch.Items[index])
		}(index)
	}
	wg.Wait()
	return results
}

// callBatch makes a batch's one call; retry says it failed for one named
// question, so its items are asked one by one.
func (e Exchange) callBatch(ctx context.Context, batch systemoneprotocol.Batch) ([]ItemResult, bool) {
	results, err := e.call(ctx, batch)
	if err != nil {
		out := make([]ItemResult, len(batch.Items))
		for index := range out {
			out[index].Err = err
		}
		return out, false
	}
	for _, result := range results {
		var problem *systemoneprotocol.Problem
		if errors.As(result.Err, &problem) && problem.Question != "" {
			return nil, true
		}
	}
	return results, false
}

// askItem asks one request: once, once more without a question the server
// names as failed, and once more after an overloaded answer's Retry-After.
func (e Exchange) askItem(ctx context.Context, batch systemoneprotocol.Batch, item systemoneprotocol.NamedRequest) ItemResult {
	request := item.Request
	var failed []systemoneprotocol.Member
	waited := false
	for {
		single := systemoneprotocol.Batch{Model: batch.Model, Options: batch.Options, Items: []systemoneprotocol.NamedRequest{{Name: item.Name, Request: request}}}
		results, err := e.call(ctx, single)
		result := ItemResult{Err: err}
		if err == nil {
			result = results[0]
		}
		var problem *systemoneprotocol.Problem
		isProblem := errors.As(result.Err, &problem)
		switch {
		case result.Err == nil:
			result.Response.Answers = append(result.Response.Answers, failed...)
			return result
		case isProblem && problem.Question != "" && len(failed) < 2 && len(request.Questions) > 1:
			remaining, questionType, found := withoutQuestion(request, problem.Question)
			if !found {
				return result
			}
			if e.Observer.Retried != nil {
				e.Observer.Retried(problem.Question, problem.QuestionError)
			}
			kind := systemoneprotocol.Null()
			if questionType != "" {
				kind = systemoneprotocol.String(questionType)
			}
			answer := systemoneprotocol.Object(
				systemoneprotocol.Member{Key: "type", Value: kind},
				systemoneprotocol.Member{Key: "error", Value: systemoneprotocol.String(firstNonEmpty(problem.QuestionError, "invalid_question"))},
			)
			if problem.Reason != "" {
				answer = answer.With("reason", systemoneprotocol.String(problem.Reason))
			}
			failed = append(failed, systemoneprotocol.Member{Key: problem.Question, Value: answer})
			request = remaining
		case isProblem && problem.Kind == systemoneprotocol.ProblemOverloaded && problem.RetryAfter > 0 && !waited && fitsDeadline(ctx, problem.RetryAfter):
			waited = true
			timer := time.NewTimer(problem.RetryAfter)
			select {
			case <-ctx.Done():
				timer.Stop()
				return ItemResult{Err: ctx.Err()}
			case <-timer.C:
			}
		default:
			return result
		}
	}
}

// call makes the one call a batch plans, its deviations applied, and reads
// its reply: one result per item, or the error without a reply.
func (e Exchange) call(ctx context.Context, batch systemoneprotocol.Batch) ([]ItemResult, error) {
	plan, diagnostics, err := e.Codec.Plan(batch, e.Target, e.Policy)
	if err != nil || len(plan.Calls) != 1 {
		if err == nil {
			err = fmt.Errorf("the plan has %d calls", len(plan.Calls))
		}
		var problem *systemoneprotocol.Problem
		if errors.As(err, &problem) {
			return nil, problem
		}
		return nil, &systemoneprotocol.Problem{Status: http.StatusBadRequest, Kind: systemoneprotocol.ProblemRejected, Detail: err.Error()}
	}
	call := plan.Calls[0]
	for _, deviation := range e.Deviations {
		rewritten, rewriteErr := deviation.RewriteRequest(&call)
		if rewriteErr != nil {
			return nil, &systemoneprotocol.Problem{Status: http.StatusBadRequest, Kind: systemoneprotocol.ProblemRejected, Detail: rewriteErr.Error()}
		}
		diagnostics = append(diagnostics, rewritten...)
	}
	if e.Observer.Diagnostics != nil && len(diagnostics) > 0 {
		e.Observer.Diagnostics(diagnostics)
	}
	body, err := json.Marshal(call.Body)
	if err != nil {
		return nil, err
	}
	started := time.Now()
	reply, err := e.Send(ctx, call, body)
	if e.Observer.Replied != nil {
		e.Observer.Replied(reply, err, time.Since(started))
	}
	if err != nil {
		return nil, err
	}
	for _, deviation := range e.Deviations {
		if rewriteErr := deviation.RewriteReply(&reply); rewriteErr != nil {
			return nil, rewriteErr
		}
	}
	results, err := e.Codec.DecodeReply(plan, call, reply)
	if err != nil {
		return nil, err
	}
	if len(results) != len(batch.Items) {
		return nil, &systemoneprotocol.Problem{Status: reply.Status, Kind: systemoneprotocol.ProblemFailed, Detail: fmt.Sprintf("the reply has %d results for %d requests", len(results), len(batch.Items))}
	}
	return results, nil
}

func withoutQuestion(request systemoneprotocol.Request, name string) (systemoneprotocol.Request, string, bool) {
	kept := make([]systemoneprotocol.NamedQuestion, 0, len(request.Questions))
	questionType, found := "", false
	for _, question := range request.Questions {
		if question.Name == name {
			questionType, found = question.Question.Type, true
			continue
		}
		kept = append(kept, question)
	}
	request.Questions = kept
	return request, questionType, found
}

func fitsDeadline(ctx context.Context, wait time.Duration) bool {
	deadline, ok := ctx.Deadline()
	return !ok || time.Until(deadline) > wait
}

// AllTransport reports whether no item got an answer from the server at all.
func AllTransport(results []ItemResult) bool {
	for _, result := range results {
		var problem *systemoneprotocol.Problem
		if result.Err == nil || errors.As(result.Err, &problem) {
			return false
		}
	}
	return len(results) > 0
}
