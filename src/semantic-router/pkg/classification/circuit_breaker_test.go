package classification

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

type fakeBackend struct {
	results chan resultOrError
}

type resultOrError struct {
	result SequenceClassificationResult
	err    error
}

func (f *fakeBackend) Classify(_ context.Context, _ string) (SequenceClassificationResult, error) {
	select {
	case r := <-f.results:
		return r.result, r.err
	default:
		return SequenceClassificationResult{}, errors.New("no result queued")
	}
}

type fakeScoringBackend struct {
	results chan scoreOrError
}

type scoreOrError struct {
	score float64
	err   error
}

func (f *fakeScoringBackend) Score(_ context.Context, _ string) (float64, error) {
	select {
	case r := <-f.results:
		return r.score, r.err
	default:
		return 0, errors.New("no result queued")
	}
}

func testCBConfig(enabled bool) *RemoteClassifierCircuitBreakerConfig {
	threshold := 3
	open := 100
	probes := 1
	return &RemoteClassifierCircuitBreakerConfig{
		Enabled:             enabled,
		ConsecutiveFailures: &threshold,
		OpenIntervalMs:      &open,
		HalfOpenMaxRequests: &probes,
	}
}

func TestCircuitBreakerClosedToOpen(t *testing.T) {
	fake := &fakeBackend{results: make(chan resultOrError, 10)}
	cfg := testCBConfig(true)
	wrapped := newCircuitBreakingBackend(fake, cfg, "test-model")

	for i := 0; i < 3; i++ {
		fake.results <- resultOrError{err: errors.New("timeout")}
		if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
			t.Fatalf("step %d: expected error", i)
		}
	}

	_, err := wrapped.Classify(context.Background(), "text")
	if err == nil {
		t.Fatal("expected circuit breaker open error")
	}
	var cbErr *ErrCircuitBreakerOpen
	if !errors.As(err, &cbErr) {
		t.Fatalf("expected *ErrCircuitBreakerOpen, got %T: %v", err, err)
	}
}

func TestCircuitBreakerSuccessResets(t *testing.T) {
	fake := &fakeBackend{results: make(chan resultOrError, 10)}
	cfg := testCBConfig(true)
	wrapped := newCircuitBreakingBackend(fake, cfg, "test-model")

	fake.results <- resultOrError{err: errors.New("timeout")}
	if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
		t.Fatal("expected error")
	}

	fake.results <- resultOrError{result: SequenceClassificationResult{Probabilities: []float32{0.9, 0.1}}}
	if _, err := wrapped.Classify(context.Background(), "text"); err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	fake.results <- resultOrError{result: SequenceClassificationResult{Probabilities: []float32{0.9, 0.1}}}
	if _, err := wrapped.Classify(context.Background(), "text"); err != nil {
		t.Fatalf("expected success, got %v", err)
	}
}

func TestCircuitBreakerDisabled(t *testing.T) {
	fake := &fakeBackend{results: make(chan resultOrError, 10)}
	cfg := testCBConfig(false)
	wrapped := newCircuitBreakingBackend(fake, cfg, "test")

	for i := 0; i < 10; i++ {
		fake.results <- resultOrError{err: errors.New("timeout")}
		if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
			t.Fatalf("expected error on call %d", i)
		}
	}
}

func TestCircuitBreakerHalfOpenToClosed(t *testing.T) {
	fake := &fakeBackend{results: make(chan resultOrError, 10)}
	threshold := 2
	openInterval := 50
	probes := 1
	cfg := &RemoteClassifierCircuitBreakerConfig{
		Enabled:             true,
		ConsecutiveFailures: &threshold,
		OpenIntervalMs:      &openInterval,
		HalfOpenMaxRequests: &probes,
	}
	wrapped := newCircuitBreakingBackend(fake, cfg, "test")

	for i := 0; i < 2; i++ {
		fake.results <- resultOrError{err: errors.New("timeout")}
		if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
			t.Fatalf("expected error %d", i)
		}
	}

	if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
		t.Fatal("expected circuit breaker open")
	}

	time.Sleep(60 * time.Millisecond)

	fake.results <- resultOrError{result: SequenceClassificationResult{Probabilities: []float32{0.9, 0.1}}}
	if _, err := wrapped.Classify(context.Background(), "text"); err != nil {
		t.Fatalf("expected half-open probe to succeed, got %v", err)
	}

	fake.results <- resultOrError{err: errors.New("timeout")}
	if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
		t.Fatal("expected error (breaker is closed but backend failed)")
	}
}

func TestCircuitBreakerHalfOpenFails(t *testing.T) {
	fake := &fakeBackend{results: make(chan resultOrError, 10)}
	threshold := 1
	openInterval := 50
	probes := 1
	cfg := &RemoteClassifierCircuitBreakerConfig{
		Enabled:             true,
		ConsecutiveFailures: &threshold,
		OpenIntervalMs:      &openInterval,
		HalfOpenMaxRequests: &probes,
	}
	wrapped := newCircuitBreakingBackend(fake, cfg, "test")

	fake.results <- resultOrError{err: errors.New("timeout")}
	if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
		t.Fatal("expected error")
	}

	time.Sleep(60 * time.Millisecond)

	fake.results <- resultOrError{err: errors.New("timeout")}
	if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
		t.Fatal("expected half-open probe failure")
	}

	if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
		t.Fatal("expected circuit breaker open after failed probe")
	}
	var cbErr *ErrCircuitBreakerOpen
	if err := wrapped.Classify(context.Background(), "text"); err != nil {
		if !errors.As(err, &cbErr) {
			t.Fatalf("expected *ErrCircuitBreakerOpen, got %T: %v", err, err)
		}
	} else {
		t.Fatal("expected circuit breaker open after failed probe")
	}
}

func TestCircuitBreakerNilConfig(t *testing.T) {
	fake := &fakeBackend{results: make(chan resultOrError, 10)}
	wrapped := newCircuitBreakingBackend(fake, nil, "test")

	for i := 0; i < 10; i++ {
		fake.results <- resultOrError{err: errors.New("timeout")}
		if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
			t.Fatalf("expected error on call %d", i)
		}
	}
}

func TestCircuitBreakerNilConfigScoringBackend(t *testing.T) {
	fake := &fakeScoringBackend{results: make(chan scoreOrError, 10)}
	wrapped := newCircuitBreakingBackendScoring(fake, nil, "test")

	fake.results <- scoreOrError{err: errors.New("timeout")}
	if _, err := wrapped.Score(context.Background(), "text"); err == nil {
		t.Fatal("expected error")
	}
}

func TestCircuitBreakerConcurrentSafety(t *testing.T) {
	fake := &fakeBackend{results: make(chan resultOrError, 100)}
	cfg := testCBConfig(true)
	wrapped := newCircuitBreakingBackend(fake, cfg, "test")

	go func() {
		for {
			fake.results <- resultOrError{result: SequenceClassificationResult{Probabilities: []float32{0.5, 0.5}}}
			time.Sleep(time.Millisecond)
		}
	}()

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = wrapped.Classify(context.Background(), "text")
		}()
	}
	wg.Wait()
}