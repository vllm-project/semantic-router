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

package selection

import (
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func emptyRatingsSnapshot() map[string]map[string]*ModelRating {
	return map[string]map[string]*ModelRating{}
}

// TestFileEloStorage_StartAutoSave_NonPositiveInterval verifies a zero or
// negative interval never reaches time.NewTicker (which panics on <= 0) and so
// cannot crash the auto-save goroutine.
func TestFileEloStorage_StartAutoSave_NonPositiveInterval(t *testing.T) {
	for _, interval := range []time.Duration{0, -time.Second} {
		storagePath := filepath.Join(t.TempDir(), "ratings.json")
		storage, err := NewFileEloStorage(storagePath)
		if err != nil {
			t.Fatalf("NewFileEloStorage: %v", err)
		}
		storage.StartAutoSave(interval, emptyRatingsSnapshot) // must not panic
		time.Sleep(20 * time.Millisecond)                     // let the goroutine run
		if err := storage.Close(); err != nil {
			t.Fatalf("Close() after StartAutoSave(%s): %v", interval, err)
		}
	}
}

// TestFileEloStorage_Close_ConcurrentSafe is a regression test locking in the
// sync.Once-based close: many goroutines closing simultaneously must not panic.
func TestFileEloStorage_Close_ConcurrentSafe(t *testing.T) {
	storagePath := filepath.Join(t.TempDir(), "ratings.json")
	storage, err := NewFileEloStorage(storagePath)
	if err != nil {
		t.Fatalf("NewFileEloStorage: %v", err)
	}
	storage.StartAutoSave(50*time.Millisecond, emptyRatingsSnapshot)

	const closers = 64
	var wg sync.WaitGroup
	start := make(chan struct{})
	wg.Add(closers)
	for i := 0; i < closers; i++ {
		go func() {
			defer wg.Done()
			<-start
			_ = storage.Close()
		}()
	}
	close(start)
	wg.Wait()
}
