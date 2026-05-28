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

// Package inflight tracks the number of concurrent in-flight chat completion
// requests per model. It is the data source for load-aware selection (e.g.
// the multi_factor selector). The tracker is self-healing: each Begin call
// records a start timestamp, and any entries older than DefaultMaxAge are
// considered abandoned and dropped from the count. This means a missed End
// (panic, lost stream, etc.) self-corrects after at most DefaultMaxAge,
// rather than leaking forever as a naive counter would.
//
// The package mirrors the pkg/latency global-state pattern intentionally so
// selectors can call simple package-level read functions without threading a
// tracker handle through SelectionContext.
package inflight

import (
	"strings"
	"sync"
	"time"
)

// DefaultMaxAge bounds how long an inflight entry is trusted before being
// treated as abandoned. Chosen to comfortably exceed the longest reasonable
// LLM streaming completion while still recovering from missed End calls in
// minutes, not days. Configurable via SetMaxAge.
const DefaultMaxAge = 10 * time.Minute

type entry struct {
	id    uint64
	start time.Time
}

type modelState struct {
	nextID  uint64
	entries map[uint64]entry
}

var (
	mu     sync.RWMutex
	states = map[string]*modelState{}
	maxAge = DefaultMaxAge
)

// SetMaxAge overrides DefaultMaxAge for the global tracker. Non-positive
// values are ignored. Exposed primarily for tests; production callers should
// rely on the default.
func SetMaxAge(d time.Duration) {
	if d <= 0 {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	maxAge = d
}

// Begin records the start of an in-flight request for model and returns a
// token that must be passed to End to clear the entry. An empty model name
// is ignored and the returned token will be a no-op when passed to End.
func Begin(model string) uint64 {
	model = strings.TrimSpace(model)
	if model == "" {
		return 0
	}
	mu.Lock()
	defer mu.Unlock()
	st, ok := states[model]
	if !ok {
		st = &modelState{entries: map[uint64]entry{}}
		states[model] = st
	}
	st.nextID++
	id := st.nextID
	st.entries[id] = entry{id: id, start: time.Now()}
	return id
}

// End clears the in-flight entry identified by (model, token). Calling End
// with token == 0 or with a token that no longer exists (already evicted
// by age or already ended) is a no-op.
func End(model string, token uint64) {
	model = strings.TrimSpace(model)
	if model == "" || token == 0 {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	st, ok := states[model]
	if !ok {
		return
	}
	delete(st.entries, token)
	if len(st.entries) == 0 {
		delete(states, model)
	}
}

// Get returns the current in-flight count for model, after evicting any
// entries older than the configured max age.
func Get(model string) int {
	model = strings.TrimSpace(model)
	if model == "" {
		return 0
	}
	mu.Lock()
	defer mu.Unlock()
	return evictAndCount(model, time.Now())
}

// Snapshot returns a copy of in-flight counts across all known models,
// post-eviction. Useful for the Prometheus gauge mirror.
func Snapshot() map[string]int {
	mu.Lock()
	defer mu.Unlock()
	now := time.Now()
	out := make(map[string]int, len(states))
	for model := range states {
		c := evictAndCount(model, now)
		if c > 0 {
			out[model] = c
		}
	}
	return out
}

// Reset clears all in-flight state. For tests only.
func Reset() {
	mu.Lock()
	defer mu.Unlock()
	states = map[string]*modelState{}
	maxAge = DefaultMaxAge
}

// evictAndCount drops abandoned entries for model and returns the remaining
// count. Caller MUST hold mu (write lock) before calling. May delete the
// model entry from states if it is empty after eviction.
func evictAndCount(model string, now time.Time) int {
	st, ok := states[model]
	if !ok {
		return 0
	}
	for id, e := range st.entries {
		if now.Sub(e.start) > maxAge {
			delete(st.entries, id)
		}
	}
	if len(st.entries) == 0 {
		delete(states, model)
		return 0
	}
	return len(st.entries)
}
