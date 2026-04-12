package ml_binding

import (
	"sync"
	"testing"
)

// TestKNNSelectorToJSONCloseRace verifies that concurrent calls to ToJSON() and
// Close() do not trigger a data race (go test -race).
func TestKNNSelectorToJSONCloseRace(t *testing.T) {
	s := NewKNNSelector(3)
	if s == nil {
		t.Skip("native library not available, skipping race test")
	}
	defer s.Close()

	const numReaders = 5
	var wg sync.WaitGroup

	for i := 0; i < numReaders; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				_, _ = s.ToJSON()
			}
		}()
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		s.Close()
	}()

	wg.Wait()
}

// TestKMeansSelectorToJSONCloseRace is the same race test for KMeansSelector.
func TestKMeansSelectorToJSONCloseRace(t *testing.T) {
	s := NewKMeansSelector(3)
	if s == nil {
		t.Skip("native library not available, skipping race test")
	}
	defer s.Close()

	const numReaders = 5
	var wg sync.WaitGroup

	for i := 0; i < numReaders; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				_, _ = s.ToJSON()
			}
		}()
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		s.Close()
	}()

	wg.Wait()
}

// TestSVMSelectorToJSONCloseRace is the same race test for SVMSelector.
func TestSVMSelectorToJSONCloseRace(t *testing.T) {
	s := NewSVMSelector()
	if s == nil {
		t.Skip("native library not available, skipping race test")
	}
	defer s.Close()

	const numReaders = 5
	var wg sync.WaitGroup

	for i := 0; i < numReaders; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				_, _ = s.ToJSON()
			}
		}()
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		s.Close()
	}()

	wg.Wait()
}
