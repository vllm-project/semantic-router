package handlers

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRunBoundedContainerCommandStopsAfterContextDeadline(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	started := time.Now()
	_, err := runBoundedCommand(ctx, "/bin/sh", 1024, "-c", "sleep 30")
	if err == nil {
		t.Fatal("expected a deadline error")
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("command returned after %s, want at most 1s", elapsed)
	}
}

func TestRunBoundedContainerCommandCapsCombinedOutput(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	output, err := runBoundedCommand(
		ctx,
		"/bin/sh",
		64,
		"-c",
		"i=0; while [ $i -lt 256 ]; do printf x; i=$((i + 1)); done",
	)
	if !errors.Is(err, errCommandOutputLimit) {
		t.Fatalf("error = %v, want output-limit error", err)
	}
	if len(output) != 64 {
		t.Fatalf("output length = %d, want 64", len(output))
	}
}
