//go:build !windows && cgo

package classification

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestUnifiedClassifier_ClassifyBatchDoesNotSerializeLegacyCalls(t *testing.T) {
	classifier := &UnifiedClassifier{
		initialized: true,
	}

	release := make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() {
		close(release)
	})
	entered := make(chan struct{}, 2)
	var current int32
	var maxConcurrent int32

	classifier.testClassifyBatchLegacy = func(texts []string) (*UnifiedBatchResults, error) {
		active := atomic.AddInt32(&current, 1)
		updateMaxInt32(&maxConcurrent, active)
		entered <- struct{}{}
		<-release
		atomic.AddInt32(&current, -1)
		return newMockUnifiedBatchResults(len(texts)), nil
	}

	errCh := make(chan error, 2)
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := classifier.ClassifyBatch([]string{"test"})
			errCh <- err
		}()
	}

	waitForConcurrentEntries(t, entered, 2)
	releaseOnce.Do(func() {
		close(release)
	})
	wg.Wait()
	close(errCh)

	for err := range errCh {
		if err != nil {
			t.Fatalf("ClassifyBatch returned error: %v", err)
		}
	}

	if maxConcurrent < 2 {
		t.Fatalf("expected concurrent legacy classification calls, got max concurrency %d", maxConcurrent)
	}
}

func TestUnifiedClassifier_ClassifyBatchInitializesLoRAOnceConcurrently(t *testing.T) {
	classifier := &UnifiedClassifier{
		initialized: true,
		useLoRA:     true,
	}

	var initCalls int32
	classifier.testInitializeLoRA = func() error {
		atomic.AddInt32(&initCalls, 1)
		time.Sleep(20 * time.Millisecond)
		return nil
	}
	classifier.testClassifyBatchWithLoRA = func(texts []string) (*UnifiedBatchResults, error) {
		return newMockUnifiedBatchResults(len(texts)), nil
	}

	errCh := make(chan error, 4)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := classifier.ClassifyBatch([]string{"test"})
			errCh <- err
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		if err != nil {
			t.Fatalf("ClassifyBatch returned error: %v", err)
		}
	}

	if initCalls != 1 {
		t.Fatalf("expected LoRA initialization once, got %d", initCalls)
	}
	if !classifier.loraInitialized {
		t.Fatal("expected classifier to mark LoRA bindings initialized")
	}
}

func newMockUnifiedBatchResults(batchSize int) *UnifiedBatchResults {
	results := &UnifiedBatchResults{
		IntentResults:   make([]IntentResult, batchSize),
		PIIResults:      make([]PIIResult, batchSize),
		SecurityResults: make([]SecurityResult, batchSize),
		BatchSize:       batchSize,
	}

	for i := 0; i < batchSize; i++ {
		results.IntentResults[i] = IntentResult{Category: "mock", Confidence: 0.9}
		results.PIIResults[i] = PIIResult{Confidence: 0.9}
		results.SecurityResults[i] = SecurityResult{Confidence: 0.9}
	}

	return results
}

func waitForConcurrentEntries(t *testing.T, entered <-chan struct{}, count int) {
	t.Helper()

	deadline := time.After(2 * time.Second)
	for i := 0; i < count; i++ {
		select {
		case <-entered:
		case <-deadline:
			t.Fatalf("timed out waiting for %d concurrent classifier entries", count)
		}
	}
}

func updateMaxInt32(target *int32, value int32) {
	for {
		current := atomic.LoadInt32(target)
		if value <= current {
			return
		}
		if atomic.CompareAndSwapInt32(target, current, value) {
			return
		}
	}
}
