/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package lookuptable_test

import (
	"sync"
	"testing"
	"time"
)

// TestFileStorage_Close_Idempotent verifies repeated sequential Close calls are
// safe no-ops and never panic on a double channel close.
func TestFileStorage_Close_Idempotent(t *testing.T) {
	s, _ := newTempFileStorage(t)
	s.StartAutoSave(50 * time.Millisecond)

	for i := 0; i < 5; i++ {
		if err := s.Close(); err != nil {
			t.Fatalf("Close() call %d returned error: %v", i, err)
		}
	}
}

// TestFileStorage_Close_ConcurrentSafe verifies many goroutines closing the
// same storage simultaneously never panic (regression for the unsafe
// check-then-close pattern that double-closed stopChan).
func TestFileStorage_Close_ConcurrentSafe(t *testing.T) {
	s, _ := newTempFileStorage(t)
	s.StartAutoSave(50 * time.Millisecond)

	const closers = 64
	var wg sync.WaitGroup
	start := make(chan struct{})
	wg.Add(closers)
	for i := 0; i < closers; i++ {
		go func() {
			defer wg.Done()
			<-start // barrier: maximise the race
			_ = s.Close()
		}()
	}
	close(start)
	wg.Wait()
}

// TestFileStorage_StartAutoSave_NonPositiveInterval verifies a zero or
// negative interval never reaches time.NewTicker (which panics on <= 0) and
// so cannot crash the auto-save goroutine.
func TestFileStorage_StartAutoSave_NonPositiveInterval(t *testing.T) {
	for _, interval := range []time.Duration{0, -time.Second} {
		s, _ := newTempFileStorage(t)
		s.StartAutoSave(interval)         // must not panic in the goroutine
		time.Sleep(20 * time.Millisecond) // give the goroutine a chance to run
		if err := s.Close(); err != nil {
			t.Fatalf("Close() after StartAutoSave(%s) returned error: %v", interval, err)
		}
	}
}

// TestFileStorage_Close_BeforeStartAutoSave verifies Close is safe when
// StartAutoSave was never called (no goroutine to join).
func TestFileStorage_Close_BeforeStartAutoSave(t *testing.T) {
	s, _ := newTempFileStorage(t)
	if err := s.Close(); err != nil {
		t.Fatalf("Close() without StartAutoSave returned error: %v", err)
	}
	// A second close must also be safe.
	if err := s.Close(); err != nil {
		t.Fatalf("second Close() returned error: %v", err)
	}
}
