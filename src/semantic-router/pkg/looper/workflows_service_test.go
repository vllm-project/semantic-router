package looper

import (
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestWorkflowStateService_AcquireAfterClose(t *testing.T) {
	cfg := &config.LooperConfig{}
	s := NewWorkflowStateService(cfg)

	if !s.Acquire() {
		t.Fatal("expected true on active service")
	}
	s.Release()

	if err := s.Close(); err != nil {
		t.Fatalf("Close error: %v", err)
	}

	if s.Acquire() {
		t.Fatal("expected false after close")
	}
}

func TestWorkflowStateService_DrainBeforeClose(t *testing.T) {
	cfg := &config.LooperConfig{}
	s := NewWorkflowStateService(cfg)

	if !s.Acquire() {
		t.Fatal("failed to acquire")
	}

	closed := make(chan struct{})
	go func() {
		_ = s.Close()
		close(closed)
	}()

	select {
	case <-closed:
		t.Fatal("Close returned before Release")
	case <-time.After(50 * time.Millisecond):
		// Expected, it is blocking on the wg.Wait()
	}

	s.Release()

	select {
	case <-closed:
		// Success
	case <-time.After(1 * time.Second):
		t.Fatal("Close did not return after Release")
	}
}

func TestWorkflowStateService_ReloadSafety(t *testing.T) {
	// simulate Swap+Close under -race
	cfg1 := &config.LooperConfig{}
	s1 := NewWorkflowStateService(cfg1)

	cfg2 := &config.LooperConfig{}
	s2 := NewWorkflowStateService(cfg2)

	// goroutine holds Acquire on old service
	if !s1.Acquire() {
		t.Fatal("failed to acquire s1")
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// Swap happens externally, we just close the old one
		_ = s1.Close()
	}()

	// New service is immediately usable
	if !s2.Acquire() {
		t.Fatal("failed to acquire s2")
	}
	s2.Release()
	_ = s2.Close()

	// s1 close is still blocked
	s1.Release()
	wg.Wait()
}

func TestWorkflowsLooper_CloseStopsSweeper(t *testing.T) {
	cfg := &config.LooperConfig{}
	// NewWorkflowsLooper creates a memory store by default, which spawns a sweeper.
	l := NewWorkflowsLooper(cfg)

	done := make(chan struct{})
	go func() {
		_ = l.Close()
		close(done)
	}()

	select {
	case <-done:
		// Success, Close waits for the sweeper to exit
	case <-time.After(2 * time.Second):
		t.Fatal("Close timed out on memory store, sweeper goroutine likely leaked")
	}
}

func TestFileStoreClose_WaitsForSweeper(t *testing.T) {
	// This test specifically verifies the file store's sweepLoop goroutine is
	// tracked and drained when Close() is called. It was previously not tracked.
	dir := t.TempDir()
	s := newWorkflowFileToolStateStore(dir, 30*time.Minute)

	done := make(chan struct{})
	go func() {
		_ = s.Close()
		close(done)
	}()

	select {
	case <-done:
		// Success, Close waited for sweepLoop to exit
	case <-time.After(2 * time.Second):
		t.Fatal("Close timed out on file store, sweeper goroutine likely leaked")
	}
}
