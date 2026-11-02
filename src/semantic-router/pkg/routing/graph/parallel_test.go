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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
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
	// Both slow hops observe the cancel, and their attempts are on the outcome
	// before Run returns. A later record would miss the run's evidence.
	if cancelled.Load() != 2 {
		t.Fatalf("%d slow hops cancelled", cancelled.Load())
	}
	for _, model := range []string{"slow1", "slow2"} {
		status, ok := attemptStatus(outcome, model)
		if !ok || status != AttemptCancelled {
			t.Fatalf("%s attempt %+v in %+v", model, status, outcome.Attempts)
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

func attemptStatus(outcome *Outcome, model string) (AttemptStatus, bool) {
	if outcome == nil {
		return "", false
	}
	for _, attempt := range outcome.Attempts {
		if attempt.Model == model {
			return attempt.Status, true
		}
	}
	return "", false
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
	outcome, err := run(t, program, caller)
	var callErr *CallError
	if !errors.As(err, &callErr) || callErr.Result.Model != "bad" {
		t.Fatalf("err %v", err)
	}
	select {
	case <-sawCancel:
	default:
		t.Fatal("the slow branch was not cancelled")
	}
	if status, ok := attemptStatus(outcome, "slow"); !ok || status != AttemptCancelled {
		t.Fatalf("slow attempt %+v in %+v", status, outcome.Attempts)
	}
}

// TestParallelNoncooperativeBranch checks a branch that never selects on ctx.
// stuck waits on a release channel. fast runs only after stuck has entered.
// Run must return before release is closed.
func TestParallelNoncooperativeBranch(t *testing.T) {
	success := func(req *HopRequest) (*HopResponse, error) {
		return completion(req.Model, req.Model, 1), nil
	}
	t.Run("first-k", func(t *testing.T) {
		for range 20 {
			outcome, err := runNoncooperative(t, context.Background(), &Parallel{
				Branches: branchesOf("stuck", "fast"), FirstK: 1,
			}, success, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got := answerText(t, outcome); got != "fast" {
				t.Fatalf("answer %q", got)
			}
			status, ok := attemptStatus(outcome, "fast")
			if !ok || status != AttemptSucceeded {
				t.Fatalf("fast attempt %+v in %+v", status, outcome.Attempts)
			}
			if _, stuck := attemptStatus(outcome, "stuck"); stuck {
				t.Fatalf("stuck was recorded before it returned: %+v", outcome.Attempts)
			}
		}
	})
	t.Run("fatal", func(t *testing.T) {
		for range 20 {
			_, err := runNoncooperative(t, context.Background(), &Parallel{
				Branches: branchesOf("stuck", "fast"),
			}, func(*HopRequest) (*HopResponse, error) {
				return &HopResponse{Status: 500, Body: []byte("{}")}, nil
			}, nil)
			var callErr *CallError
			if !errors.As(err, &callErr) || callErr.Result.Model != "fast" {
				t.Fatalf("err %v", err)
			}
		}
	})
	t.Run("parent_cancel", func(t *testing.T) {
		for range 20 {
			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)
			outcome, err := runNoncooperative(t, ctx, &Parallel{
				Branches: branchesOf("stuck", "fast"), FirstK: 1,
			}, success, cancel)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("parent: %v", err)
			}
			status, ok := attemptStatus(outcome, "fast")
			if !ok || status != AttemptSucceeded {
				t.Fatalf("fast attempt %+v in %+v", status, outcome.Attempts)
			}
			if _, stuck := attemptStatus(outcome, "stuck"); stuck {
				t.Fatalf("stuck was recorded before it returned: %+v", outcome.Attempts)
			}
		}
	})
}

// runNoncooperative starts one parallel step and the following respond step.
// onFastReturned, when set, runs after fast has returned and the step context
// is done, so FirstK can record errEnough before the caller is canceled.
func runNoncooperative(t *testing.T, ctx context.Context, step *Parallel, answer func(*HopRequest) (*HopResponse, error), onFastReturned context.CancelFunc) (*Outcome, error) {
	t.Helper()
	entered := make(chan struct{})
	release := make(chan struct{})
	fastDone := make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	var enteredOnce sync.Once
	caller := &fakeCaller{answer: func(_ context.Context, req *HopRequest) (*HopResponse, error) {
		if req.Model == "stuck" {
			enteredOnce.Do(func() { close(entered) })
			<-release
			return completion(req.Model, "stuck", 1), nil
		}
		<-entered
		defer close(fastDone)
		return answer(req)
	}}
	if onFastReturned != nil {
		step.afterStop = func() {
			<-fastDone
			onFastReturned()
		}
	}
	program := &Program{Steps: Sequence{parallel("panel", step), respond()}}
	// Build the run on this goroutine. Run itself is started separately so a
	// branch that ignores cancellation cannot block the test.
	input := Input{Request: chatRequest(t, "hello")}
	options := Options{Caller: caller, Hop: routing.Hop{Decision: "panel", Recipe: "default"}}
	type result struct {
		outcome *Outcome
		err     error
	}
	done := make(chan result, 1)
	go func() {
		outcome, err := Run(ctx, program, input, options)
		done <- result{outcome, err}
	}()
	select {
	case got := <-done:
		return got.outcome, got.err
	case <-time.After(5 * time.Second):
		t.Fatal("parallel step waited for a branch that ignores cancellation")
	}
	return nil, nil
}
