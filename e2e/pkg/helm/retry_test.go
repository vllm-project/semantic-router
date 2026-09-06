package helm

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRunWithRetrySucceedsAfterTransientFailures(t *testing.T) {
	attempts := 0
	err := runWithRetry(context.Background(), 4, 0, func() error {
		attempts++
		if attempts < 3 {
			return errors.New("transient failure")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("runWithRetry returned error: %v", err)
	}
	if attempts != 3 {
		t.Fatalf("expected 3 attempts, got %d", attempts)
	}
}

func TestRunWithRetryReturnsFinalError(t *testing.T) {
	want := errors.New("final failure")
	attempts := 0
	err := runWithRetry(context.Background(), 3, 0, func() error {
		attempts++
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("expected final error %v, got %v", want, err)
	}
	if attempts != 3 {
		t.Fatalf("expected 3 attempts, got %d", attempts)
	}
}

func TestRunWithRetryStopsWhenContextIsCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	attempts := 0
	err := runWithRetry(ctx, 4, time.Hour, func() error {
		attempts++
		cancel()
		return errors.New("transient failure")
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context cancellation, got %v", err)
	}
	if attempts != 1 {
		t.Fatalf("expected 1 attempt, got %d", attempts)
	}
}

func TestIsTransientHelmInstallFailure(t *testing.T) {
	for _, output := range []string{
		`failed to perform "Fetch": response status code 500`,
		"TLS handshake timeout",
		"read: connection reset by peer",
		"unexpected EOF",
	} {
		if !isTransientHelmInstallFailure(output) {
			t.Errorf("expected transient Helm failure for %q", output)
		}
	}
	if isTransientHelmInstallFailure("template: values.yaml: field is required") {
		t.Fatal("configuration failures must not be retried")
	}
}
