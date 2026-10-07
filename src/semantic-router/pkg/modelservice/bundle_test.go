package modelservice

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestBundleSendsOneCallForEveryParticipant(t *testing.T) {
	runtime := &surfaceRuntime{}
	client := newSurfaceClient(t, runtime)
	ctx, bundle := WithBundle(context.Background(), time.Second)
	leaves := []func(){bundle.Join(), bundle.Join(), bundle.Join()}
	var wait sync.WaitGroup
	errs := make([]error, 3)
	labels := make([]string, 3)
	wait.Add(3)
	go func() {
		defer wait.Done()
		defer leaves[0]()
		response, err := client.Classify(ctx, "vela-domain", ClassifyRequest{Inputs: []ClassifyInput{{Text: "x"}}})
		errs[0] = err
		if err == nil {
			labels[0] = response.Results[0].Label
		}
	}()
	go func() {
		defer wait.Done()
		defer leaves[1]()
		_, errs[1] = client.Embed(ctx, "vela-embedding", EmbedRequest{Inputs: []EmbedInput{{Text: "x"}, {Text: "y"}}})
	}()
	go func() {
		defer wait.Done()
		defer leaves[2]()
		_, errs[2] = client.Rerank(ctx, "vela-reranker", RerankRequest{Query: "q", Documents: []string{"a", "b"}})
	}()
	wait.Wait()
	for index, err := range errs {
		if err != nil {
			t.Fatalf("participant %d: %v", index, err)
		}
	}
	if labels[0] != "billing" {
		t.Fatalf("the classify participant got %q", labels[0])
	}
	if runtime.bundles.Load() != 1 || runtime.tasks.Load() != 3 || runtime.direct.Load() != 0 {
		t.Fatalf("bundles=%d tasks=%d direct=%d", runtime.bundles.Load(), runtime.tasks.Load(), runtime.direct.Load())
	}
	if bundle.Flushes() != 1 {
		t.Fatalf("flushes=%d", bundle.Flushes())
	}
}

func TestBundleFlushesAfterTheWindowWhenAParticipantNeverCalls(t *testing.T) {
	runtime := &surfaceRuntime{}
	client := newSurfaceClient(t, runtime)
	ctx, bundle := WithBundle(context.Background(), 5*time.Millisecond)
	defer bundle.Join()()
	leave := bundle.Join()
	defer leave()
	started := time.Now()
	if _, err := client.Classify(ctx, "m", ClassifyRequest{Inputs: []ClassifyInput{{Text: "x"}}}); err != nil {
		t.Fatal(err)
	}
	if waited := time.Since(started); waited > 500*time.Millisecond {
		t.Fatalf("the call waited %s for a participant that never called", waited)
	}
}

func TestBundleFlushesWhenTheOtherParticipantsLeave(t *testing.T) {
	runtime := &surfaceRuntime{}
	client := newSurfaceClient(t, runtime)
	ctx, bundle := WithBundle(context.Background(), time.Hour)
	leaveCaller := bundle.Join()
	leaveIdle := bundle.Join()
	done := make(chan error, 1)
	go func() {
		defer leaveCaller()
		_, err := client.Classify(ctx, "m", ClassifyRequest{Inputs: []ClassifyInput{{Text: "x"}}})
		done <- err
	}()
	time.Sleep(20 * time.Millisecond)
	select {
	case err := <-done:
		t.Fatalf("the call returned before the bundle flushed: %v", err)
	default:
	}
	leaveIdle()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("leaving did not flush the bundle")
	}
}

func TestBundleSendsOneCallPerRuntime(t *testing.T) {
	first, second := &surfaceRuntime{}, &surfaceRuntime{}
	clients := []*Client{newSurfaceClient(t, first), newSurfaceClient(t, second)}
	ctx, bundle := WithBundle(context.Background(), time.Second)
	var wait sync.WaitGroup
	for _, client := range clients {
		leave := bundle.Join()
		wait.Add(1)
		go func(client *Client) {
			defer wait.Done()
			defer leave()
			if _, err := client.Classify(ctx, "m", ClassifyRequest{Inputs: []ClassifyInput{{Text: "x"}}}); err != nil {
				t.Error(err)
			}
		}(client)
	}
	wait.Wait()
	if first.bundles.Load() != 1 || second.bundles.Load() != 1 {
		t.Fatalf("bundles: %d and %d", first.bundles.Load(), second.bundles.Load())
	}
}

func TestBundleSplitsAStageAtTheRuntimeTaskCap(t *testing.T) {
	runtime := &surfaceRuntime{maxTasks: DefaultBundleTasks}
	client := newSurfaceClient(t, runtime)
	ctx, bundle := WithBundle(context.Background(), time.Second)
	leave := bundle.Join()
	calls := 2*DefaultBundleTasks + 2
	errs := make([]error, calls)
	Fan(ctx, calls, func(i int) {
		_, errs[i] = client.Classify(ctx, "vela-pii", ClassifyRequest{Inputs: []ClassifyInput{{Text: "x"}}})
	})
	leave()
	for index, err := range errs {
		if err != nil {
			t.Fatalf("call %d: %v", index, err)
		}
	}
	if runtime.bundles.Load() != 3 || runtime.tasks.Load() != int64(calls) || bundle.Flushes() != 1 {
		t.Fatalf("bundles=%d tasks=%d flushes=%d, want one flush in 3 bundles of at most %d tasks", runtime.bundles.Load(), runtime.tasks.Load(), bundle.Flushes(), DefaultBundleTasks)
	}
}

func TestBundleSplitsAtTheCapTheRuntimeReports(t *testing.T) {
	runtime := &surfaceRuntime{maxTasks: 3}
	client := newSurfaceClient(t, runtime)
	if _, err := client.Models(context.Background()); err != nil {
		t.Fatal(err)
	}
	ctx, bundle := WithBundle(context.Background(), time.Second)
	leave := bundle.Join()
	errs := make([]error, 7)
	Fan(ctx, len(errs), func(i int) {
		_, errs[i] = client.Classify(ctx, "vela-pii", ClassifyRequest{Inputs: []ClassifyInput{{Text: "x"}}})
	})
	leave()
	for index, err := range errs {
		if err != nil {
			t.Fatalf("call %d: %v", index, err)
		}
	}
	if runtime.bundles.Load() != 3 || runtime.tasks.Load() != 7 {
		t.Fatalf("bundles=%d tasks=%d, want 7 tasks in 3 bundles of at most 3", runtime.bundles.Load(), runtime.tasks.Load())
	}
}

func TestBundleTaskErrorsStayWithTheirCaller(t *testing.T) {
	client := newSurfaceClient(t, &surfaceRuntime{})
	ctx, bundle := WithBundle(context.Background(), time.Second)
	var wait sync.WaitGroup
	errs := make([]error, 2)
	for index, model := range []string{"vela-domain", "busy"} {
		leave := bundle.Join()
		wait.Add(1)
		go func(index int, model string) {
			defer wait.Done()
			defer leave()
			_, errs[index] = client.Classify(ctx, model, ClassifyRequest{Inputs: []ClassifyInput{{Text: "x"}}})
		}(index, model)
	}
	wait.Wait()
	if errs[0] != nil || !errors.Is(errs[1], ErrOverloaded) {
		t.Fatalf("got %v and %v", errs[0], errs[1])
	}
}

func TestABundledCallerStopsAtItsOwnDeadline(t *testing.T) {
	client := newSurfaceClient(t, &surfaceRuntime{delay: 300 * time.Millisecond})
	base, bundle := WithBundle(context.Background(), time.Millisecond)
	defer bundle.Join()()
	ctx, cancel := context.WithTimeout(base, 30*time.Millisecond)
	defer cancel()
	started := time.Now()
	_, err := client.Classify(ctx, "m", ClassifyRequest{Inputs: []ClassifyInput{{Text: "x"}}})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v", err)
	}
	if waited := time.Since(started); waited > 250*time.Millisecond {
		t.Fatalf("the caller waited %s past its deadline", waited)
	}
}
