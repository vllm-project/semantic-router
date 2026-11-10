package modelruntime

import (
	"context"
	"errors"
	"testing"
	"time"
)

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
