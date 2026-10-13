package graph

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestParallelCapsConcurrencyAndStartsInOrder(t *testing.T) {
	var running, peak atomic.Int32
	caller := &fakeCaller{answer: func(_ context.Context, req *HopRequest) (*HopResponse, error) {
		now := running.Add(1)
		for {
			old := peak.Load()
			if now <= old || peak.CompareAndSwap(old, now) {
				break
			}
		}
		time.Sleep(5 * time.Millisecond)
		running.Add(-1)
		return completion(req.Model, req.Model, 1), nil
	}}
	models := []string{"a", "b", "c", "d", "e"}
	program := &Program{Steps: Sequence{
		parallel("panel", &Parallel{Branches: branchesOf(models...), MaxConcurrency: 2}),
		aggregate("join", Concat{Separator: ","}),
		respond(),
	}}
	outcome, err := run(t, program, caller)
	if err != nil {
		t.Fatal(err)
	}
	if peak.Load() > 2 {
		t.Fatalf("peak concurrency %d", peak.Load())
	}
	if got := answerText(t, outcome); got != "a,b,c,d,e" {
		t.Fatalf("results out of branch order: %q", got)
	}

	serial := &fakeCaller{}
	program.Steps[0] = parallel("panel", &Parallel{Branches: branchesOf(models...), MaxConcurrency: 1})
	if _, err := run(t, program, serial); err != nil || !slices.Equal(serial.models(), models) {
		t.Fatalf("one at a time: err %v order %v", err, serial.models())
	}
}

func TestParallelFirstKCancelsTheRest(t *testing.T) {
	var cancelled atomic.Int32
	var inFlight sync.WaitGroup
	inFlight.Add(2)
	caller := &fakeCaller{answer: func(ctx context.Context, req *HopRequest) (*HopResponse, error) {
		switch req.Model {
		case "fast1", "fast2":
			inFlight.Wait()
			return completion(req.Model, req.Model, 1), nil
		}
		inFlight.Done()
		<-ctx.Done()
		cancelled.Add(1)
		return nil, ctx.Err()
	}}
	program := &Program{Steps: Sequence{
		parallel("panel", &Parallel{Branches: branchesOf("slow1", "fast1", "slow2", "fast2"), FirstK: 2}),
		aggregate("join", Concat{Separator: "+"}),
		respond(),
	}}
	outcome, err := run(t, program, caller)
	if err != nil {
		t.Fatal(err)
	}
	if got := answerText(t, outcome); got != "fast1+fast2" {
		t.Fatalf("answer %q", got)
	}
	if cancelled.Load() != 2 {
		t.Fatalf("%d slow hops cancelled", cancelled.Load())
	}
	for _, attempt := range outcome.Attempts {
		if (attempt.Model == "slow1" || attempt.Model == "slow2") && attempt.Status != AttemptCancelled {
			t.Fatalf("attempt %+v", attempt)
		}
	}
}

func TestParallelFirstKNeverStartsQueuedBranches(t *testing.T) {
	caller := &fakeCaller{}
	program := &Program{Steps: Sequence{
		parallel("panel", &Parallel{Branches: branchesOf("a", "b", "c", "d"), FirstK: 1, MaxConcurrency: 1}),
		respond(),
	}}
	if _, err := run(t, program, caller); err != nil || !slices.Equal(caller.models(), []string{"a"}) {
		t.Fatalf("err %v sent %v", err, caller.models())
	}
}

func failing(models ...string) *fakeCaller {
	return &fakeCaller{answer: func(_ context.Context, req *HopRequest) (*HopResponse, error) {
		if slices.Contains(models, req.Model) {
			return &HopResponse{Status: 503, Body: []byte("unavailable")}, nil
		}
		return completion(req.Model, req.Model, 1), nil
	}}
}

func TestParallelSkipKeepsTheSuccesses(t *testing.T) {
	program := &Program{Steps: Sequence{
		parallel("panel", &Parallel{Branches: branchesOf("a", "b", "c"), OnError: OnErrorSkip, MinSuccess: 2}),
		aggregate("choices", Choices{}),
		respond(),
	}}
	outcome, err := run(t, program, failing("b"))
	if err != nil {
		t.Fatal(err)
	}
	if body := string(outcome.Response.Answer.Body); !strings.Contains(body, `"content":"a"`) || !strings.Contains(body, `"content":"c"`) {
		t.Fatalf("body %s", body)
	}
	_, err = run(t, program, failing("a", "b"))
	var callErr *CallError
	if !errors.Is(err, ErrTooFewResults) || !errors.As(err, &callErr) || callErr.Result.Status != 503 {
		t.Fatalf("too few: %v", err)
	}
}

func TestParallelFailStopsAtTheFirstFailure(t *testing.T) {
	var once sync.Once
	sawCancel := make(chan struct{})
	slowInFlight := make(chan struct{})
	caller := &fakeCaller{answer: func(ctx context.Context, req *HopRequest) (*HopResponse, error) {
		if req.Model == "bad" {
			<-slowInFlight
			return &HopResponse{Status: 500, Body: []byte("{}")}, nil
		}
		close(slowInFlight)
		<-ctx.Done()
		once.Do(func() { close(sawCancel) })
		return nil, ctx.Err()
	}}
	program := &Program{Steps: Sequence{parallel("panel", &Parallel{Branches: branchesOf("slow", "bad")}), respond()}}
	_, err := run(t, program, caller)
	var callErr *CallError
	if !errors.As(err, &callErr) || callErr.Result.Model != "bad" {
		t.Fatalf("err %v", err)
	}
	select {
	case <-sawCancel:
	case <-time.After(5 * time.Second):
		t.Fatal("the slow branch was not cancelled")
	}
}
