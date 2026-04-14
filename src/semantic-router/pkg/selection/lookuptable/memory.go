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

package lookuptable

import "sync"

// MemoryStorage is an in-memory LookupTableStorage implementation.
// It is primarily useful for testing and short-lived processes.
// Load and Save are no-ops.
type MemoryStorage struct {
	mu      sync.RWMutex
	entries map[string]Entry
}

// NewMemoryStorage creates an empty in-memory storage backend.
func NewMemoryStorage() *MemoryStorage {
	return &MemoryStorage{
		entries: make(map[string]Entry),
	}
}

// Get implements LookupTable.
func (m *MemoryStorage) Get(key Key) (Entry, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	e, ok := m.entries[key.String()]
	return e, ok
}

// QualityGap implements LookupTable.
func (m *MemoryStorage) QualityGap(taskFamily, currentModel, candidateModel string) (float64, bool) {
	e, ok := m.Get(QualityGapKey(taskFamily, currentModel, candidateModel))
	return e.Value, ok
}

// HandoffPenalty implements LookupTable.
func (m *MemoryStorage) HandoffPenalty(fromModel, toModel string) (float64, bool) {
	e, ok := m.Get(HandoffPenaltyKey(fromModel, toModel))
	return e.Value, ok
}

// RemainingTurnPrior implements LookupTable.
func (m *MemoryStorage) RemainingTurnPrior(intentOrDomain string) (float64, bool) {
	e, ok := m.Get(RemainingTurnPriorKey(intentOrDomain))
	return e.Value, ok
}

// Set implements LookupTableStorage.
func (m *MemoryStorage) Set(key Key, entry Entry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries[key.String()] = entry
	return nil
}

// All implements LookupTableStorage.
func (m *MemoryStorage) All() map[string]Entry {
	m.mu.RLock()
	defer m.mu.RUnlock()
	cp := make(map[string]Entry, len(m.entries))
	for k, v := range m.entries {
		cp[k] = v
	}
	return cp
}

// Load is a no-op for in-memory storage.
func (m *MemoryStorage) Load() error { return nil }

// Save is a no-op for in-memory storage.
func (m *MemoryStorage) Save() error { return nil }

// Close is a no-op for in-memory storage.
func (m *MemoryStorage) Close() error { return nil }
