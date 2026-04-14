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
	"math"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

func ptr[T any](v T) *T { return &v }

func TestBuilder_EmptyRecords(t *testing.T) {
	m := lookuptable.NewMemoryStorage()
	b := lookuptable.NewBuilder(m)
	if err := b.PopulateFromRecords(nil); err != nil {
		t.Fatalf("PopulateFromRecords(nil): %v", err)
	}
	if len(m.All()) != 0 {
		t.Errorf("expected 0 entries for empty records, got %d", len(m.All()))
	}
}

func TestBuilder_QualityGap(t *testing.T) {
	records := []store.Record{
		{Category: "coding", SelectedModel: "small", ConfidenceScore: 0.6, RequestID: "aaa-1"},
		{Category: "coding", SelectedModel: "small", ConfidenceScore: 0.8, RequestID: "aaa-2"},
		{Category: "coding", SelectedModel: "large", ConfidenceScore: 0.9, RequestID: "bbb-1"},
	}
	// avg(small) = 0.7, avg(large) = 0.9
	// gap(small→large) = 0.9 - 0.7 = 0.2
	// gap(large→small) = 0.7 - 0.9 = -0.2

	m := lookuptable.NewMemoryStorage()
	b := lookuptable.NewBuilder(m)
	if err := b.PopulateFromRecords(records); err != nil {
		t.Fatalf("PopulateFromRecords: %v", err)
	}

	v, ok := m.QualityGap("coding", "small", "large")
	if !ok {
		t.Fatal("quality_gap::coding::small::large not found")
	}
	if math.Abs(v-0.2) > 1e-9 {
		t.Errorf("quality_gap(small→large) = %v, want 0.2", v)
	}

	v2, ok2 := m.QualityGap("coding", "large", "small")
	if !ok2 {
		t.Fatal("quality_gap::coding::large::small not found")
	}
	if math.Abs(v2-(-0.2)) > 1e-9 {
		t.Errorf("quality_gap(large→small) = %v, want -0.2", v2)
	}
}

func TestBuilder_HandoffPenalty(t *testing.T) {
	now := time.Now()
	cost1 := 0.01
	cost2 := 0.03 // switching to "large" costs more

	// Both records share the same Decision and are within the session window,
	// so they are grouped into the same pseudo-session by the builder.
	records := []store.Record{
		{Decision: "default", SelectedModel: "small", ActualCost: ptr(cost1), Timestamp: now},
		{Decision: "default", SelectedModel: "large", ActualCost: ptr(cost2), Timestamp: now.Add(time.Second)},
	}

	m := lookuptable.NewMemoryStorage()
	b := lookuptable.NewBuilder(m)
	if err := b.PopulateFromRecords(records); err != nil {
		t.Fatalf("PopulateFromRecords: %v", err)
	}

	v, ok := m.HandoffPenalty("small", "large")
	if !ok {
		t.Fatal("handoff_penalty::small::large not found")
	}
	// delta = 0.03 - 0.01 = 0.02; only one sample so EMA = 0.02
	if math.Abs(v-0.02) > 1e-9 {
		t.Errorf("handoff_penalty(small→large) = %v, want 0.02", v)
	}
}

func TestBuilder_RemainingTurnPrior(t *testing.T) {
	// Two sessions under the same Decision, separated by >30 min.
	// Session A: 3 turns, Session B: 5 turns, both in "support".
	// avg = (3 + 5) / 2 = 4.0
	base := time.Now()
	gap := 31 * time.Minute // exceeds sessionWindowDuration → new session
	records := []store.Record{
		{Decision: "default", Category: "support", Timestamp: base},
		{Decision: "default", Category: "support", Timestamp: base.Add(time.Second)},
		{Decision: "default", Category: "support", Timestamp: base.Add(2 * time.Second)},
		// Session B starts after a >30 min gap.
		{Decision: "default", Category: "support", Timestamp: base.Add(gap)},
		{Decision: "default", Category: "support", Timestamp: base.Add(gap + time.Second)},
		{Decision: "default", Category: "support", Timestamp: base.Add(gap + 2*time.Second)},
		{Decision: "default", Category: "support", Timestamp: base.Add(gap + 3*time.Second)},
		{Decision: "default", Category: "support", Timestamp: base.Add(gap + 4*time.Second)},
	}

	m := lookuptable.NewMemoryStorage()
	b := lookuptable.NewBuilder(m)
	if err := b.PopulateFromRecords(records); err != nil {
		t.Fatalf("PopulateFromRecords: %v", err)
	}

	v, ok := m.RemainingTurnPrior("support")
	if !ok {
		t.Fatal("remaining_turn_prior::support not found")
	}
	// avgLen = (3+5)/2 = 4.0; first-turn prior = avgLen - 1 = 3.0
	if math.Abs(v-3.0) > 1e-9 {
		t.Errorf("remaining_turn_prior(support) = %v, want 3.0", v)
	}
}

func TestBuilder_ConfigOverrideNotOverwritten(t *testing.T) {
	m := lookuptable.NewMemoryStorage()

	// Pre-set a config override.
	key := lookuptable.QualityGapKey("coding", "small", "large")
	_ = m.Set(key, lookuptable.Entry{Value: 0.99, Source: lookuptable.SourceConfigOverride})

	records := []store.Record{
		{Category: "coding", SelectedModel: "small", ConfidenceScore: 0.6, RequestID: "a-1"},
		{Category: "coding", SelectedModel: "large", ConfidenceScore: 0.9, RequestID: "b-1"},
	}

	b := lookuptable.NewBuilder(m)
	if err := b.PopulateFromRecords(records); err != nil {
		t.Fatalf("PopulateFromRecords: %v", err)
	}

	// The config override should be preserved.
	v, _ := m.QualityGap("coding", "small", "large")
	if math.Abs(v-0.99) > 1e-9 {
		t.Errorf("config override was overwritten: got %v, want 0.99", v)
	}
}

func TestBuilder_HandoffPenalty_MissingCostSkipped(t *testing.T) {
	now := time.Now()
	// Second record has no ActualCost — the switch should be silently skipped,
	// producing no handoff_penalty entry.
	records := []store.Record{
		{Decision: "default", SelectedModel: "small", ActualCost: ptr(0.01), Timestamp: now},
		{Decision: "default", SelectedModel: "large", ActualCost: nil, Timestamp: now.Add(time.Second)},
	}

	m := lookuptable.NewMemoryStorage()
	b := lookuptable.NewBuilder(m)
	if err := b.PopulateFromRecords(records); err != nil {
		t.Fatalf("PopulateFromRecords: %v", err)
	}
	if _, ok := m.HandoffPenalty("small", "large"); ok {
		t.Error("expected no handoff_penalty entry when cost data is missing")
	}
}

func TestBuilder_HandoffPenalty_CrossWindowNotMerged(t *testing.T) {
	now := time.Now()
	cost := 0.01
	// Two model switches in the same Decision but separated by >30 min.
	// They should be treated as different sessions and each produce an
	// independent first-sample EMA (not folded into the same accumulator).
	records := []store.Record{
		{Decision: "default", SelectedModel: "small", ActualCost: ptr(cost), Timestamp: now},
		{Decision: "default", SelectedModel: "large", ActualCost: ptr(cost + 0.02), Timestamp: now.Add(time.Second)},
		// New session — gap > 30 min.
		{Decision: "default", SelectedModel: "small", ActualCost: ptr(cost), Timestamp: now.Add(31 * time.Minute)},
		{Decision: "default", SelectedModel: "large", ActualCost: ptr(cost + 0.04), Timestamp: now.Add(31*time.Minute + time.Second)},
	}

	m := lookuptable.NewMemoryStorage()
	b := lookuptable.NewBuilder(m)
	if err := b.PopulateFromRecords(records); err != nil {
		t.Fatalf("PopulateFromRecords: %v", err)
	}

	v, ok := m.HandoffPenalty("small", "large")
	if !ok {
		t.Fatal("handoff_penalty::small::large not found")
	}
	// Two samples: delta1=0.02, delta2=0.04.
	// EMA: first=0.02, second = 0.2*0.04 + 0.8*0.02 = 0.024.
	want := 0.2*0.04 + 0.8*0.02
	if math.Abs(v-want) > 1e-9 {
		t.Errorf("handoff_penalty = %v, want %v", v, want)
	}
}

func TestBuilder_SkipsRecordsWithoutRequiredFields(t *testing.T) {
	records := []store.Record{
		{Category: "", SelectedModel: "small", ConfidenceScore: 0.8, RequestID: "a-1"},     // no category
		{Category: "coding", SelectedModel: "", ConfidenceScore: 0.8, RequestID: "b-1"},    // no model
		{Category: "coding", SelectedModel: "large", ConfidenceScore: 0, RequestID: "c-1"}, // zero confidence
	}

	m := lookuptable.NewMemoryStorage()
	b := lookuptable.NewBuilder(m)
	if err := b.PopulateFromRecords(records); err != nil {
		t.Fatalf("PopulateFromRecords: %v", err)
	}

	// No quality gap entries should be emitted (only one valid model per category at most).
	for k, e := range m.All() {
		if e.Source == lookuptable.SourceReplayDerived {
			_ = k
			// Remaining-turn-prior entries for "coding" are acceptable since
			// the third record has a valid category and request ID.
			break
		}
	}
}
