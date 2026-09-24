package modelruntime

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestWarmupRouterRunsReadyTasksAndSkipsUnavailableTasks(t *testing.T) {
	loaded := make([]string, 0, 2)
	summary, err := WarmupRouter(context.Background(), []RouterWarmupTask{
		{
			Name:  "knowledge_bases",
			Ready: true,
			Load: func() error {
				loaded = append(loaded, "knowledge_bases")
				return nil
			},
		},
		{
			Name:       "tools_database",
			Ready:      false,
			SkipReason: "embedding runtime unavailable",
			Load: func() error {
				t.Fatal("unready warmup task must not run")
				return nil
			},
		},
	}, WarmupRouterOptions{Component: "test", MaxParallelism: 2})
	if err != nil {
		t.Fatalf("WarmupRouter() error = %v", err)
	}
	if len(loaded) != 1 || loaded[0] != "knowledge_bases" {
		t.Fatalf("loaded tasks = %v, want knowledge_bases", loaded)
	}
	if result := summary.Results["router.warmup.knowledge_bases"]; result.Status != TaskSucceeded {
		t.Fatalf("knowledge base warmup result = %+v, want success", result)
	}
}

func TestWarmupRouterTreatsTaskFailureAsBestEffort(t *testing.T) {
	summary, err := WarmupRouter(context.Background(), []RouterWarmupTask{{
		Name:  "knowledge_bases",
		Ready: true,
		Load:  func() error { return errors.New("embedding failed") },
	}}, WarmupRouterOptions{Component: "test", MaxParallelism: 1})
	if err != nil {
		t.Fatalf("WarmupRouter() best-effort error = %v", err)
	}
	if result := summary.Results["router.warmup.knowledge_bases"]; result.Status != TaskFailed {
		t.Fatalf("knowledge base warmup result = %+v, want failure", result)
	}
}

func TestExecuteReturnsWhenStartupCancellationFindsAStuckTask(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := Execute(ctx, []Task{{
			Name: "stuck-startup-task",
			Run: func(context.Context) error {
				close(started)
				<-release
				return nil
			},
		}}, Options{MaxParallelism: 1})
		done <- err
	}()

	<-started
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Execute() error = %v, want context canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Execute() did not return after startup cancellation")
	}
}

func TestWarmupRouterDoesNotStartLoadAfterCancellation(t *testing.T) {
	loadCalled := make(chan struct{}, 1)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := WarmupRouter(ctx, []RouterWarmupTask{{
		Name:  "cancelled-warmup",
		Ready: true,
		Load: func() error {
			loadCalled <- struct{}{}
			return nil
		},
	}}, WarmupRouterOptions{MaxParallelism: 1})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("WarmupRouter() error = %v, want context canceled", err)
	}
	select {
	case <-loadCalled:
		t.Fatal("WarmupRouter() called Load after cancellation")
	case <-time.After(25 * time.Millisecond):
	}
}
