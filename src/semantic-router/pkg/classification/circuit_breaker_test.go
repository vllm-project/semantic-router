package classification

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
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

// connTransportError builds a connector error that isUnavailableError counts.
func connTransportError() error {
	return &connector.Error{Kind: connector.KindTransport, Operation: "http_classify", Retryable: true, Cause: errors.New("connection refused")}
}

// connCancelledError builds a connector error wrapping a caller cancellation,
// which the connector produces with KindTransport and Retryable=false.
func connCancelledError() error {
	return &connector.Error{Kind: connector.KindTransport, Operation: "http_classify", Retryable: false, Cause: context.Canceled}
}

// connNonRetryableTransportError builds a connector transport failure that is
// not retryable and therefore must not count toward the breaker.
func connNonRetryableTransportError() error {
	return &connector.Error{Kind: connector.KindTransport, Operation: "http_classify", Retryable: false, Cause: errors.New("connection reset")}
}

func testCBConfig(enabled bool) *config.RemoteClassifierCircuitBreakerConfig {
	threshold := 3
	open := 100
	probes := 1
	return &config.RemoteClassifierCircuitBreakerConfig{
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
		fake.results <- resultOrError{err: connTransportError()}
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

	fake.results <- resultOrError{err: connTransportError()}
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
		fake.results <- resultOrError{err: connTransportError()}
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
	cfg := &config.RemoteClassifierCircuitBreakerConfig{
		Enabled:             true,
		ConsecutiveFailures: &threshold,
		OpenIntervalMs:      &openInterval,
		HalfOpenMaxRequests: &probes,
	}
	wrapped := newCircuitBreakingBackend(fake, cfg, "test")

	for i := 0; i < 2; i++ {
		fake.results <- resultOrError{err: connTransportError()}
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

	fake.results <- resultOrError{err: connTransportError()}
	if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
		t.Fatal("expected error (breaker is closed but backend failed)")
	}
}

func TestCircuitBreakerHalfOpenFails(t *testing.T) {
	fake := &fakeBackend{results: make(chan resultOrError, 10)}
	threshold := 1
	openInterval := 50
	probes := 1
	cfg := &config.RemoteClassifierCircuitBreakerConfig{
		Enabled:             true,
		ConsecutiveFailures: &threshold,
		OpenIntervalMs:      &openInterval,
		HalfOpenMaxRequests: &probes,
	}
	wrapped := newCircuitBreakingBackend(fake, cfg, "test")

	fake.results <- resultOrError{err: connTransportError()}
	if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
		t.Fatal("expected error")
	}

	time.Sleep(60 * time.Millisecond)

	fake.results <- resultOrError{err: connTransportError()}
	if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
		t.Fatal("expected half-open probe failure")
	}

	if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
		t.Fatal("expected circuit breaker open after failed probe")
	}
	var cbErr *ErrCircuitBreakerOpen
	if _, err := wrapped.Classify(context.Background(), "text"); err != nil {
		if !errors.As(err, &cbErr) {
			t.Fatalf("expected *ErrCircuitBreakerOpen, got %T: %v", err, err)
		}
	} else {
		t.Fatal("expected circuit breaker open after failed probe")
	}
}

// TestCircuitBreakerConnectorCancellationNotCounted verifies that a retryable
// connector error wrapping a cancelled context does not count toward the breaker
// (the connector sets Retryable=false for cancelled contexts).
func TestCircuitBreakerConnectorCancellationNotCounted(t *testing.T) {
	fake := &fakeBackend{results: make(chan resultOrError, 10)}
	cfg := testCBConfig(true)
	wrapped := newCircuitBreakingBackend(fake, cfg, "test-model")

	// Two connector-wrapped cancellations must not count toward threshold=3.
	for i := 0; i < 2; i++ {
		fake.results <- resultOrError{err: connCancelledError()}
		_, err := wrapped.Classify(context.Background(), "text")
		if err == nil {
			t.Fatalf("step %d: expected error", i)
		}
	}

	// Three transport errors = threshold, should open breaker.
	for i := 0; i < 3; i++ {
		fake.results <- resultOrError{err: connTransportError()}
		_, err := wrapped.Classify(context.Background(), "text")
		if err == nil {
			t.Fatalf("expected error on transport failure %d", i)
		}
	}

	// Next call should be rejected by the open breaker.
	_, err := wrapped.Classify(context.Background(), "text")
	var cbErr *ErrCircuitBreakerOpen
	if !errors.As(err, &cbErr) {
		t.Fatalf("expected *ErrCircuitBreakerOpen after open, got %T: %v", err, err)
	}
}

// TestCircuitBreakerNonRetryableTransportNotCounted verifies that a
// non-retryable transport failure (Retryable=false) does not count toward the
// breaker (item 1).
func TestCircuitBreakerNonRetryableTransportNotCounted(t *testing.T) {
	fake := &fakeBackend{results: make(chan resultOrError, 10)}
	cfg := testCBConfig(true)
	wrapped := newCircuitBreakingBackend(fake, cfg, "test-model")

	// Two non-retryable transport errors must not count toward threshold=3.
	for i := 0; i < 2; i++ {
		fake.results <- resultOrError{err: connNonRetryableTransportError()}
		_, err := wrapped.Classify(context.Background(), "text")
		if err == nil {
			t.Fatalf("step %d: expected error", i)
		}
	}

	// Three retryable transport errors = threshold, should open breaker.
	for i := 0; i < 3; i++ {
		fake.results <- resultOrError{err: connTransportError()}
		_, err := wrapped.Classify(context.Background(), "text")
		if err == nil {
			t.Fatalf("expected error on transport failure %d", i)
		}
	}

	_, err := wrapped.Classify(context.Background(), "text")
	var cbErr *ErrCircuitBreakerOpen
	if !errors.As(err, &cbErr) {
		t.Fatalf("expected *ErrCircuitBreakerOpen after open, got %T: %v", err, err)
	}
}

// TestHalfOpenNonCountedErrorProbeReleases verifies that an admitted half-open
// probe returning a non-counted error (e.g. caller cancellation) does not leave
// the breaker stuck in half-open (item 4). The probe slot must be released and
// the half-open round must resolve, so subsequent requests are not skipped.
func TestHalfOpenNonCountedErrorProbeReleases(t *testing.T) {
	fake := &fakeBackend{results: make(chan resultOrError, 10)}
	threshold := 1
	openInterval := 50
	probes := 1
	cfg := &config.RemoteClassifierCircuitBreakerConfig{
		Enabled:             true,
		ConsecutiveFailures: &threshold,
		OpenIntervalMs:      &openInterval,
		HalfOpenMaxRequests: &probes,
	}
	wrapped := newCircuitBreakingBackend(fake, cfg, "test")

	// Trip the breaker.
	fake.results <- resultOrError{err: connTransportError()}
	if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
		t.Fatal("expected error")
	}

	// Wait for half-open interval.
	time.Sleep(60 * time.Millisecond)

	// Half-open probe returns a non-counted error (connector cancellation).
	// This must NOT leave the breaker stuck — instead the slot is released
	// and the half-open round resolves (no counted failures → closed).
	fake.results <- resultOrError{err: connCancelledError()}
	_, err := wrapped.Classify(context.Background(), "text")
	if err == nil {
		t.Fatal("expected error from cancelled half-open probe")
	}

	// The breaker should now be closed and admit the next request to the backend.
	fake.results <- resultOrError{result: SequenceClassificationResult{Probabilities: []float32{0.9, 0.1}}}
	if _, err := wrapped.Classify(context.Background(), "text"); err != nil {
		t.Fatalf("expected success after half-open cancellation release, got %v", err)
	}
}

// TestHalfOpenMaxProbesGreaterThanOne verifies that with maxProbes > 1 the
// breaker remains half-open until all admitted probes resolve, reopening on
// any failure (item 3).
func TestHalfOpenMaxProbesGreaterThanOne(t *testing.T) {
	threshold := 3
	openInterval := 50
	probes := 2
	cfg := &config.RemoteClassifierCircuitBreakerConfig{
		Enabled:             true,
		ConsecutiveFailures: &threshold,
		OpenIntervalMs:      &openInterval,
		HalfOpenMaxRequests: &probes,
	}
	cb := newCircuitBreaker("test", cfg)

	// Trip breaker.
	for i := 0; i < 3; i++ {
		cb.recordFailure()
	}
	if got := cb.getState(); got != circuitBreakerOpen {
		t.Fatalf("expected open after threshold, got %v", got)
	}

	// Wait for half-open interval.
	time.Sleep(60 * time.Millisecond)

	// Admit two probes.
	if !cb.allow() {
		t.Fatal("expected to admit probe 1")
	}
	if !cb.allow() {
		t.Fatal("expected to admit probe 2")
	}

	// First probe succeeds — must NOT close the breaker (other probe unresolved).
	cb.recordSuccess()
	if got := cb.getState(); got != circuitBreakerHalfOpen {
		t.Fatalf("expected half-open after first success (pending probe), got %v", got)
	}

	// Second probe fails — with maxProbes > 1 the breaker records the failure
	// while keeping the leftover success as neutral. All unresolved now
	// resolved, halfOpenFailed=true → open.
	cb.recordFailure()
	if got := cb.getState(); got != circuitBreakerOpen {
		t.Fatalf("expected open after failure in half-open with maxProbes=2, got %v", got)
	}

	// Subsequent requests must be rejected.
	if cb.allow() {
		t.Fatal("expected reject after open from half-open failure")
	}
}

// TestCircuitBreakerGaugeTransitions verifies the prometheus state gauge is
// updated on each state transition (item 5).
func TestCircuitBreakerGaugeTransitions(t *testing.T) {
	threshold := 2
	openInterval := 50
	probes := 1
	cfg := &config.RemoteClassifierCircuitBreakerConfig{
		Enabled:             true,
		ConsecutiveFailures: &threshold,
		OpenIntervalMs:      &openInterval,
		HalfOpenMaxRequests: &probes,
	}
	name := "gauge-test"

	cb := newCircuitBreaker(name, cfg)
	assertGauge(t, name, 0) // closed

	cb.recordFailure()
	assertGauge(t, name, 0) // still closed (failureCount=1 < threshold=2)

	cb.recordFailure()
	assertGauge(t, name, 1) // open

	time.Sleep(60 * time.Millisecond)

	cb.allow()              // open→half-open
	assertGauge(t, name, 2) // half-open

	cb.recordSuccess()      // half-open→closed
	assertGauge(t, name, 0) // closed
}

func assertGauge(t *testing.T, name string, want float64) {
	t.Helper()
	got := testutil.ToFloat64(circuitBreakerStateGauge.WithLabelValues(name))
	if got != want {
		t.Errorf("gauge(%q) = %f, want %f", name, got, want)
	}
}

func TestCircuitBreakerNilConfig(t *testing.T) {
	fake := &fakeBackend{results: make(chan resultOrError, 10)}
	wrapped := newCircuitBreakingBackend(fake, nil, "test")

	for i := 0; i < 10; i++ {
		fake.results <- resultOrError{err: connTransportError()}
		if _, err := wrapped.Classify(context.Background(), "text"); err == nil {
			t.Fatalf("expected error on call %d", i)
		}
	}
}

func TestCircuitBreakerNilConfigScoringBackend(t *testing.T) {
	fake := &fakeScoringBackend{results: make(chan scoreOrError, 10)}
	wrapped := newCircuitBreakingBackendScoring(fake, nil, "test")

	fake.results <- scoreOrError{err: connTransportError()}
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
