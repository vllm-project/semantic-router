/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package inflight

import (
	"sync"
	"testing"
	"time"
)

func TestBeginEndSymmetric(t *testing.T) {
	Reset()
	tok := Begin("m1")
	if tok == 0 {
		t.Fatal("Begin returned zero token for valid model")
	}
	if got := Get("m1"); got != 1 {
		t.Errorf("after Begin: count = %d, want 1", got)
	}
	End("m1", tok)
	if got := Get("m1"); got != 0 {
		t.Errorf("after End: count = %d, want 0", got)
	}
}

func TestMultipleInflightSameModel(t *testing.T) {
	Reset()
	tokens := []uint64{Begin("m1"), Begin("m1"), Begin("m1")}
	if got := Get("m1"); got != 3 {
		t.Errorf("3 begins: count = %d, want 3", got)
	}
	End("m1", tokens[1])
	if got := Get("m1"); got != 2 {
		t.Errorf("after one End: count = %d, want 2", got)
	}
	End("m1", tokens[0])
	End("m1", tokens[2])
	if got := Get("m1"); got != 0 {
		t.Errorf("after all Ends: count = %d, want 0", got)
	}
}

func TestEmptyModelIgnored(t *testing.T) {
	Reset()
	if tok := Begin(""); tok != 0 {
		t.Errorf("Begin(\"\") = %d, want 0", tok)
	}
	if got := Get(""); got != 0 {
		t.Errorf("Get(\"\") = %d, want 0", got)
	}
	End("", 1)
}

func TestZeroTokenEndIsNoop(t *testing.T) {
	Reset()
	tok := Begin("m1")
	End("m1", 0)
	if got := Get("m1"); got != 1 {
		t.Errorf("End with token=0 should be no-op, count = %d, want 1", got)
	}
	End("m1", tok)
	End("m1", tok)
}

func TestSelfHealingViaMaxAge(t *testing.T) {
	Reset()
	SetMaxAge(50 * time.Millisecond)
	Begin("m1")
	Begin("m1")
	if got := Get("m1"); got != 2 {
		t.Fatalf("initial count = %d, want 2", got)
	}
	time.Sleep(100 * time.Millisecond)
	if got := Get("m1"); got != 0 {
		t.Errorf("after maxAge elapsed: count = %d, want 0 (abandoned entries evicted)", got)
	}
}

func TestSnapshotPerModel(t *testing.T) {
	Reset()
	Begin("m1")
	Begin("m1")
	Begin("m2")
	snap := Snapshot()
	if snap["m1"] != 2 || snap["m2"] != 1 {
		t.Errorf("snapshot = %v, want m1=2 m2=1", snap)
	}
	if _, ok := snap["m3"]; ok {
		t.Errorf("snapshot should not contain m3")
	}
}

func TestConcurrentBeginEnd(t *testing.T) {
	Reset()
	const goroutines = 50
	const perGoroutine = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < perGoroutine; i++ {
				tok := Begin("hot-model")
				End("hot-model", tok)
			}
		}()
	}
	wg.Wait()
	if got := Get("hot-model"); got != 0 {
		t.Errorf("after symmetric Begin/End from %d goroutines: count = %d, want 0", goroutines, got)
	}
}
