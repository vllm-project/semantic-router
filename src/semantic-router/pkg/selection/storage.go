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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// EloStorage defines the interface for persisting Elo ratings
// This allows ratings to survive restarts and enables different storage backends
type EloStorage interface {
	// LoadRatings loads ratings for a specific category from storage
	// Returns empty map if category doesn't exist (not an error)
	LoadRatings(category string) (map[string]*ModelRating, error)

	// SaveRatings persists ratings for a specific category
	SaveRatings(category string, ratings map[string]*ModelRating) error

	// LoadAllRatings loads all ratings across all categories
	// Returns a map of category -> model -> rating
	LoadAllRatings() (map[string]map[string]*ModelRating, error)

	// SaveAllRatings persists all ratings across all categories
	SaveAllRatings(ratings map[string]map[string]*ModelRating) error

	// Close releases any resources held by the storage backend
	Close() error
}

// StoredRatings represents the JSON structure for persisted ratings
type StoredRatings struct {
	// Version for future format migrations
	Version int `json:"version"`

	// LastUpdated timestamp
	LastUpdated time.Time `json:"last_updated"`

	// GlobalRatings are category-independent ratings
	GlobalRatings map[string]*ModelRating `json:"global_ratings"`

	// CategoryRatings are per-category ratings (category -> model -> rating)
	CategoryRatings map[string]map[string]*ModelRating `json:"category_ratings"`
}

// FileEloStorage implements EloStorage using a JSON file
type FileEloStorage struct {
	path     string
	mu       sync.RWMutex
	dirty    bool
	stopChan chan struct{}
	doneChan chan struct{}
}

// NewFileEloStorage creates a new file-based storage backend
// The path should be a writable file path (e.g., "/var/lib/vsr/elo_ratings.json")
func NewFileEloStorage(path string) (*FileEloStorage, error) {
	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create storage directory: %w", err)
	}

	storage := &FileEloStorage{
		path:     path,
		stopChan: make(chan struct{}),
		doneChan: make(chan struct{}),
	}

	logging.Infof("[EloStorage] Initialized file storage: %s", path)
	return storage, nil
}

// LoadRatings loads ratings for a specific category
func (f *FileEloStorage) LoadRatings(category string) (map[string]*ModelRating, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	stored, err := f.loadFromFile()
	if err != nil {
		return nil, err
	}

	if category == "" || category == "_global" {
		return stored.GlobalRatings, nil
	}

	if ratings, ok := stored.CategoryRatings[category]; ok {
		return ratings, nil
	}

	return make(map[string]*ModelRating), nil
}

// SaveRatings saves ratings for a specific category
func (f *FileEloStorage) SaveRatings(category string, ratings map[string]*ModelRating) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	stored, err := f.loadFromFile()
	if err != nil {
		// If file doesn't exist, create new structure
		stored = &StoredRatings{
			Version:         1,
			GlobalRatings:   make(map[string]*ModelRating),
			CategoryRatings: make(map[string]map[string]*ModelRating),
		}
	}

	if category == "" || category == "_global" {
		stored.GlobalRatings = ratings
	} else {
		if stored.CategoryRatings == nil {
			stored.CategoryRatings = make(map[string]map[string]*ModelRating)
		}
		stored.CategoryRatings[category] = ratings
	}

	stored.LastUpdated = time.Now()

	return f.saveToFile(stored)
}

// LoadAllRatings loads all ratings from storage
func (f *FileEloStorage) LoadAllRatings() (map[string]map[string]*ModelRating, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	stored, err := f.loadFromFile()
	if err != nil {
		return nil, err
	}

	result := make(map[string]map[string]*ModelRating)

	// Add global ratings under special key
	if len(stored.GlobalRatings) > 0 {
		result["_global"] = stored.GlobalRatings
	}

	// Add category ratings
	for cat, ratings := range stored.CategoryRatings {
		result[cat] = ratings
	}

	return result, nil
}

// SaveAllRatings saves all ratings to storage
func (f *FileEloStorage) SaveAllRatings(ratings map[string]map[string]*ModelRating) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	stored := &StoredRatings{
		Version:         1,
		LastUpdated:     time.Now(),
		GlobalRatings:   make(map[string]*ModelRating),
		CategoryRatings: make(map[string]map[string]*ModelRating),
	}

	for cat, catRatings := range ratings {
		if cat == "_global" {
			stored.GlobalRatings = catRatings
		} else {
			stored.CategoryRatings[cat] = catRatings
		}
	}

	return f.saveToFile(stored)
}

// Close stops background operations and releases resources
func (f *FileEloStorage) Close() error {
	close(f.stopChan)
	<-f.doneChan
	logging.Infof("[EloStorage] Closed file storage: %s", f.path)
	return nil
}

// loadFromFile reads the storage file
func (f *FileEloStorage) loadFromFile() (*StoredRatings, error) {
	data, err := os.ReadFile(f.path)
	if err != nil {
		if os.IsNotExist(err) {
			// Return empty structure if file doesn't exist
			return &StoredRatings{
				Version:         1,
				GlobalRatings:   make(map[string]*ModelRating),
				CategoryRatings: make(map[string]map[string]*ModelRating),
			}, nil
		}
		return nil, fmt.Errorf("failed to read storage file: %w", err)
	}

	var stored StoredRatings
	if err := json.Unmarshal(data, &stored); err != nil {
		return nil, fmt.Errorf("failed to parse storage file: %w", err)
	}

	// Initialize nil maps
	if stored.GlobalRatings == nil {
		stored.GlobalRatings = make(map[string]*ModelRating)
	}
	if stored.CategoryRatings == nil {
		stored.CategoryRatings = make(map[string]map[string]*ModelRating)
	}

	return &stored, nil
}

// saveToFile writes the storage file atomically
func (f *FileEloStorage) saveToFile(stored *StoredRatings) error {
	data, err := json.MarshalIndent(stored, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal ratings: %w", err)
	}

	// Write to temp file first for atomic operation
	tmpPath := f.path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write temp file: %w", err)
	}

	// Atomic rename
	if err := os.Rename(tmpPath, f.path); err != nil {
		// Cleanup temp file on failure
		os.Remove(tmpPath)
		return fmt.Errorf("failed to rename temp file: %w", err)
	}

	logging.Debugf("[EloStorage] Saved ratings to %s", f.path)
	return nil
}

// StartAutoSave starts a background goroutine that periodically saves dirty ratings
func (f *FileEloStorage) StartAutoSave(interval time.Duration, getAll func() map[string]map[string]*ModelRating) {
	go func() {
		defer close(f.doneChan)

		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-f.stopChan:
				// Final save before shutdown
				if f.dirty {
					ratings := getAll()
					if err := f.SaveAllRatings(ratings); err != nil {
						logging.Errorf("[EloStorage] Failed final save: %v", err)
					}
				}
				return
			case <-ticker.C:
				if f.dirty {
					ratings := getAll()
					if err := f.SaveAllRatings(ratings); err != nil {
						logging.Errorf("[EloStorage] Auto-save failed: %v", err)
					} else {
						f.dirty = false
					}
				}
			}
		}
	}()
}

// MarkDirty marks that ratings have changed and need to be saved
func (f *FileEloStorage) MarkDirty() {
	f.dirty = true
}

// MemoryEloStorage implements EloStorage using in-memory storage (for testing)
type MemoryEloStorage struct {
	mu       sync.RWMutex
	global   map[string]*ModelRating
	category map[string]map[string]*ModelRating
}

// NewMemoryEloStorage creates a new in-memory storage backend
func NewMemoryEloStorage() *MemoryEloStorage {
	return &MemoryEloStorage{
		global:   make(map[string]*ModelRating),
		category: make(map[string]map[string]*ModelRating),
	}
}

// LoadRatings loads ratings for a specific category
func (m *MemoryEloStorage) LoadRatings(category string) (map[string]*ModelRating, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if category == "" || category == "_global" {
		result := make(map[string]*ModelRating)
		for k, v := range m.global {
			result[k] = v
		}
		return result, nil
	}

	if ratings, ok := m.category[category]; ok {
		result := make(map[string]*ModelRating)
		for k, v := range ratings {
			result[k] = v
		}
		return result, nil
	}

	return make(map[string]*ModelRating), nil
}

// SaveRatings saves ratings for a specific category
func (m *MemoryEloStorage) SaveRatings(category string, ratings map[string]*ModelRating) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if category == "" || category == "_global" {
		m.global = make(map[string]*ModelRating)
		for k, v := range ratings {
			m.global[k] = v
		}
	} else {
		if m.category == nil {
			m.category = make(map[string]map[string]*ModelRating)
		}
		m.category[category] = make(map[string]*ModelRating)
		for k, v := range ratings {
			m.category[category][k] = v
		}
	}

	return nil
}

// LoadAllRatings loads all ratings from storage
func (m *MemoryEloStorage) LoadAllRatings() (map[string]map[string]*ModelRating, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]map[string]*ModelRating)

	if len(m.global) > 0 {
		result["_global"] = make(map[string]*ModelRating)
		for k, v := range m.global {
			result["_global"][k] = v
		}
	}

	for cat, ratings := range m.category {
		result[cat] = make(map[string]*ModelRating)
		for k, v := range ratings {
			result[cat][k] = v
		}
	}

	return result, nil
}

// SaveAllRatings saves all ratings to storage
func (m *MemoryEloStorage) SaveAllRatings(ratings map[string]map[string]*ModelRating) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.global = make(map[string]*ModelRating)
	m.category = make(map[string]map[string]*ModelRating)

	for cat, catRatings := range ratings {
		if cat == "_global" {
			for k, v := range catRatings {
				m.global[k] = v
			}
		} else {
			m.category[cat] = make(map[string]*ModelRating)
			for k, v := range catRatings {
				m.category[cat][k] = v
			}
		}
	}

	return nil
}

// Close is a no-op for memory storage
func (m *MemoryEloStorage) Close() error {
	return nil
}
