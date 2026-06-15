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
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

func TestMemoryStorage_SetGet(t *testing.T) {
	m := lookuptable.NewMemoryStorage()
	key := lookuptable.QualityGapKey("coding", "gpt-4", "claude-3")
	entry := lookuptable.Entry{Value: 0.15, Source: lookuptable.SourceManual, UpdatedAt: time.Now()}

	if err := m.Set(key, entry); err != nil {
		t.Fatalf("Set: %v", err)
	}

	got, ok := m.Get(key)
	if !ok {
		t.Fatal("Get: entry not found after Set")
	}
	if got.Value != entry.Value {
		t.Errorf("Get.Value = %v, want %v", got.Value, entry.Value)
	}
}

func TestMemoryStorage_GetMissing(t *testing.T) {
	m := lookuptable.NewMemoryStorage()
	_, ok := m.Get(lookuptable.QualityGapKey("coding", "a", "b"))
	if ok {
		t.Error("Get on empty storage should return ok=false")
	}
}

func TestMemoryStorage_ConvenienceMethods(t *testing.T) {
	m := lookuptable.NewMemoryStorage()
	_ = m.Set(lookuptable.QualityGapKey("coding", "gpt-4", "claude-3"), lookuptable.Entry{Value: 0.15})
	_ = m.Set(lookuptable.HandoffPenaltyKey("gpt-4", "claude-3"), lookuptable.Entry{Value: 0.07})
	_ = m.Set(lookuptable.RemainingTurnPriorKey("support"), lookuptable.Entry{Value: 4.2})

	if v, ok := m.QualityGap("coding", "gpt-4", "claude-3"); !ok || v != 0.15 {
		t.Errorf("QualityGap = (%v, %v), want (0.15, true)", v, ok)
	}
	if v, ok := m.HandoffPenalty("gpt-4", "claude-3"); !ok || v != 0.07 {
		t.Errorf("HandoffPenalty = (%v, %v), want (0.07, true)", v, ok)
	}
	if v, ok := m.RemainingTurnPrior("support"); !ok || v != 4.2 {
		t.Errorf("RemainingTurnPrior = (%v, %v), want (4.2, true)", v, ok)
	}
}

func TestMemoryStorage_AllIsACopy(t *testing.T) {
	m := lookuptable.NewMemoryStorage()
	_ = m.Set(lookuptable.QualityGapKey("x", "a", "b"), lookuptable.Entry{Value: 1.0})

	all := m.All()
	// Mutate the copy; the storage should be unaffected.
	for k := range all {
		delete(all, k)
	}

	if len(m.All()) != 1 {
		t.Error("All() returned a reference; mutating it affected the storage")
	}
}

func TestMemoryStorage_LoadSaveClose(t *testing.T) {
	m := lookuptable.NewMemoryStorage()
	if err := m.Load(); err != nil {
		t.Errorf("Load should be a no-op, got: %v", err)
	}
	if err := m.Save(); err != nil {
		t.Errorf("Save should be a no-op, got: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Errorf("Close should be a no-op, got: %v", err)
	}
}
