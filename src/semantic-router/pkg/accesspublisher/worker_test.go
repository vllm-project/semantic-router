package accesspublisher

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

type workerProcessor struct {
	mu      sync.Mutex
	results []ProcessResult
	errors  []error
	calls   int
}

func (processor *workerProcessor) ProcessOnce(context.Context) (ProcessResult, error) {
	processor.mu.Lock()
	defer processor.mu.Unlock()
	index := processor.calls
	processor.calls++
	if index < len(processor.errors) && processor.errors[index] != nil {
		return ProcessResult{}, processor.errors[index]
	}
	if index < len(processor.results) {
		return processor.results[index], nil
	}
	return ProcessResult{Disposition: ProcessNoWork}, nil
}

func (processor *workerProcessor) count() int {
	processor.mu.Lock()
	defer processor.mu.Unlock()
	return processor.calls
}

func TestWorkerRecoversReadinessAfterTransientFailure(t *testing.T) {
	processor := &workerProcessor{
		errors:  []error{errors.New("Valkey unavailable")},
		results: []ProcessResult{{}, {Disposition: ProcessApplied}},
	}
	worker, err := NewWorker(WorkerOptions{
		Processor: processor, IdleDelay: time.Millisecond,
		MinBackoff: time.Millisecond, MaxBackoff: 2 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := worker.Ready(context.Background()); err == nil {
		t.Fatal("worker reported ready before Run")
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- worker.Run(ctx) }()
	deadline := time.Now().Add(time.Second)
	for processor.count() < 3 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if processor.count() < 3 {
		t.Fatal("worker did not retry the failed publication")
	}
	if err := worker.Ready(context.Background()); err != nil {
		t.Fatalf("worker did not recover readiness: %v", err)
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("Run() = %v, want context cancellation", err)
	}
}

func TestWorkerWaitsWhenNoPublicationExists(t *testing.T) {
	processor := &workerProcessor{}
	worker, err := NewWorker(WorkerOptions{
		Processor: processor, IdleDelay: 10 * time.Millisecond,
		MinBackoff: time.Millisecond, MaxBackoff: 2 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- worker.Run(ctx) }()
	time.Sleep(25 * time.Millisecond)
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("Run() = %v, want context cancellation", err)
	}
	if calls := processor.count(); calls < 2 || calls > 5 {
		t.Fatalf("idle processor calls = %d, want bounded polling", calls)
	}
}
