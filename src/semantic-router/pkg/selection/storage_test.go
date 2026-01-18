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
	"os"
	"path/filepath"
	"testing"
)

func TestFileEloStorage_SaveAndLoad(t *testing.T) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "elo-storage-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	storagePath := filepath.Join(tmpDir, "ratings.json")

	// Create storage
	storage, err := NewFileEloStorage(storagePath)
	if err != nil {
		t.Fatalf("Failed to create storage: %v", err)
	}

	// Test saving global ratings
	globalRatings := map[string]*ModelRating{
		"model-a": {Model: "model-a", Rating: 1550.0, Wins: 10, Losses: 5},
		"model-b": {Model: "model-b", Rating: 1450.0, Wins: 5, Losses: 10},
	}

	err = storage.SaveRatings("_global", globalRatings)
	if err != nil {
		t.Fatalf("Failed to save global ratings: %v", err)
	}

	// Test saving category ratings
	categoryRatings := map[string]*ModelRating{
		"model-a": {Model: "model-a", Rating: 1600.0, Wins: 15, Losses: 3},
		"model-b": {Model: "model-b", Rating: 1400.0, Wins: 3, Losses: 15},
	}

	err = storage.SaveRatings("code", categoryRatings)
	if err != nil {
		t.Fatalf("Failed to save category ratings: %v", err)
	}

	// Create new storage instance to test loading
	storage2, err := NewFileEloStorage(storagePath)
	if err != nil {
		t.Fatalf("Failed to create second storage: %v", err)
	}

	// Load and verify global ratings
	loadedGlobal, err := storage2.LoadRatings("_global")
	if err != nil {
		t.Fatalf("Failed to load global ratings: %v", err)
	}

	if len(loadedGlobal) != 2 {
		t.Errorf("Expected 2 global ratings, got %d", len(loadedGlobal))
	}

	if loadedGlobal["model-a"].Rating != 1550.0 {
		t.Errorf("Expected model-a rating 1550.0, got %f", loadedGlobal["model-a"].Rating)
	}

	// Load and verify category ratings
	loadedCategory, err := storage2.LoadRatings("code")
	if err != nil {
		t.Fatalf("Failed to load category ratings: %v", err)
	}

	if len(loadedCategory) != 2 {
		t.Errorf("Expected 2 category ratings, got %d", len(loadedCategory))
	}

	if loadedCategory["model-a"].Rating != 1600.0 {
		t.Errorf("Expected model-a category rating 1600.0, got %f", loadedCategory["model-a"].Rating)
	}
}

func TestFileEloStorage_LoadNonExistent(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "elo-storage-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	storagePath := filepath.Join(tmpDir, "nonexistent.json")

	storage, err := NewFileEloStorage(storagePath)
	if err != nil {
		t.Fatalf("Failed to create storage: %v", err)
	}

	// Loading from non-existent file should return empty map
	ratings, err := storage.LoadRatings("_global")
	if err != nil {
		t.Fatalf("Unexpected error loading from non-existent file: %v", err)
	}

	if len(ratings) != 0 {
		t.Errorf("Expected empty ratings, got %d", len(ratings))
	}
}

func TestFileEloStorage_SaveAllAndLoadAll(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "elo-storage-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	storagePath := filepath.Join(tmpDir, "ratings.json")

	storage, err := NewFileEloStorage(storagePath)
	if err != nil {
		t.Fatalf("Failed to create storage: %v", err)
	}

	// Create ratings for multiple categories
	allRatings := map[string]map[string]*ModelRating{
		"_global": {
			"model-a": {Model: "model-a", Rating: 1500.0},
		},
		"code": {
			"model-a": {Model: "model-a", Rating: 1600.0},
			"model-b": {Model: "model-b", Rating: 1400.0},
		},
		"chat": {
			"model-b": {Model: "model-b", Rating: 1550.0},
		},
	}

	err = storage.SaveAllRatings(allRatings)
	if err != nil {
		t.Fatalf("Failed to save all ratings: %v", err)
	}

	// Load all ratings
	loaded, err := storage.LoadAllRatings()
	if err != nil {
		t.Fatalf("Failed to load all ratings: %v", err)
	}

	if len(loaded) != 3 {
		t.Errorf("Expected 3 categories, got %d", len(loaded))
	}

	if len(loaded["code"]) != 2 {
		t.Errorf("Expected 2 models in code category, got %d", len(loaded["code"]))
	}
}

func TestMemoryEloStorage(t *testing.T) {
	storage := NewMemoryEloStorage()

	// Test save and load
	ratings := map[string]*ModelRating{
		"model-a": {Model: "model-a", Rating: 1550.0},
	}

	err := storage.SaveRatings("test", ratings)
	if err != nil {
		t.Fatalf("Failed to save: %v", err)
	}

	loaded, err := storage.LoadRatings("test")
	if err != nil {
		t.Fatalf("Failed to load: %v", err)
	}

	if len(loaded) != 1 {
		t.Errorf("Expected 1 rating, got %d", len(loaded))
	}

	if loaded["model-a"].Rating != 1550.0 {
		t.Errorf("Expected rating 1550.0, got %f", loaded["model-a"].Rating)
	}
}

func TestEloSelector_WithStorage(t *testing.T) {
	// Create memory storage
	storage := NewMemoryEloStorage()

	// Pre-populate storage with ratings
	storage.SaveRatings("_global", map[string]*ModelRating{
		"model-a": {Model: "model-a", Rating: 1600.0, Wins: 20, Losses: 5},
		"model-b": {Model: "model-b", Rating: 1400.0, Wins: 5, Losses: 20},
	})

	// Create config with storage
	cfg := &EloConfig{
		InitialRating:    DefaultEloRating,
		KFactor:          EloKFactor,
		CategoryWeighted: true,
		MinComparisons:   5,
	}

	// Create selector and set storage
	selector := NewEloSelector(cfg)
	selector.SetStorage(storage)

	// Load from storage
	err := selector.loadFromStorage()
	if err != nil {
		t.Fatalf("Failed to load from storage: %v", err)
	}

	// Verify ratings were loaded
	rating := selector.getGlobalRating("model-a")
	if rating == nil {
		t.Fatal("Expected model-a rating, got nil")
	}

	if rating.Rating != 1600.0 {
		t.Errorf("Expected rating 1600.0, got %f", rating.Rating)
	}
}
