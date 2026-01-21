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
	defer os.RemoveAll(tmpDir) //nolint:errcheck // cleanup in test

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
	defer os.RemoveAll(tmpDir) //nolint:errcheck // cleanup in test

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
	defer os.RemoveAll(tmpDir) //nolint:errcheck // cleanup in test

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
	if err := storage.SaveRatings("_global", map[string]*ModelRating{
		"model-a": {Model: "model-a", Rating: 1600.0, Wins: 20, Losses: 5},
		"model-b": {Model: "model-b", Rating: 1400.0, Wins: 5, Losses: 20},
	}); err != nil {
		t.Fatalf("Failed to save ratings: %v", err)
	}

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

func TestFileEloStorage_EmptyFile(t *testing.T) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "elo-storage-test-empty")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir) //nolint:errcheck // cleanup in test

	storagePath := filepath.Join(tmpDir, "empty.json")

	// Create empty file
	if writeErr := os.WriteFile(storagePath, []byte{}, 0o644); writeErr != nil {
		t.Fatalf("Failed to create empty file: %v", writeErr)
	}

	// Create storage - should not fail on empty file
	storage, err := NewFileEloStorage(storagePath)
	if err != nil {
		t.Fatalf("Failed to create storage with empty file: %v", err)
	}

	// Load should return empty ratings, not error
	ratings, err := storage.LoadRatings("_global")
	if err != nil {
		t.Errorf("Expected no error on empty file, got: %v", err)
	}

	if len(ratings) != 0 {
		t.Errorf("Expected empty ratings, got %d", len(ratings))
	}
}

func TestFileEloStorage_CorruptedFile(t *testing.T) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "elo-storage-test-corrupt")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir) //nolint:errcheck // cleanup in test

	storagePath := filepath.Join(tmpDir, "corrupted.json")

	// Create corrupted JSON file
	if writeErr := os.WriteFile(storagePath, []byte("not valid json {{{"), 0o644); writeErr != nil {
		t.Fatalf("Failed to create corrupted file: %v", writeErr)
	}

	// Create storage (this succeeds - loading is lazy)
	storage, err := NewFileEloStorage(storagePath)
	if err != nil {
		t.Fatalf("Failed to create storage: %v", err)
	}

	// LoadRatings should fail on corrupted file
	_, err = storage.LoadRatings("_global")
	if err == nil {
		t.Error("Expected error when loading corrupted file, got nil")
	}

	// Check that backup was created
	backupPath := storagePath + ".corrupted"
	if _, statErr := os.Stat(backupPath); os.IsNotExist(statErr) {
		t.Error("Expected backup file to be created")
	}
}

func TestFileEloStorage_InvalidRatings(t *testing.T) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "elo-storage-test-invalid")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir) //nolint:errcheck // cleanup in test

	storagePath := filepath.Join(tmpDir, "invalid.json")

	// Create file with out-of-bounds ratings
	invalidJSON := `{
		"version": 1,
		"global_ratings": {
			"model-a": {"model": "model-a", "rating": -500, "wins": -10, "losses": 5},
			"model-b": {"model": "model-b", "rating": 5000, "wins": 10, "losses": -5}
		},
		"category_ratings": {}
	}`
	if writeErr := os.WriteFile(storagePath, []byte(invalidJSON), 0o644); writeErr != nil {
		t.Fatalf("Failed to create invalid file: %v", writeErr)
	}

	// Create storage
	storage, err := NewFileEloStorage(storagePath)
	if err != nil {
		t.Fatalf("Failed to create storage: %v", err)
	}

	// Load ratings - should succeed with sanitized values
	ratings, err := storage.LoadRatings("_global")
	if err != nil {
		t.Fatalf("Failed to load ratings: %v", err)
	}

	// Check that ratings were clamped
	if ratings["model-a"].Rating != 100 {
		t.Errorf("Expected model-a rating to be clamped to 100, got %f", ratings["model-a"].Rating)
	}
	if ratings["model-b"].Rating != 3000 {
		t.Errorf("Expected model-b rating to be clamped to 3000, got %f", ratings["model-b"].Rating)
	}

	// Check that negative wins/losses were fixed
	if ratings["model-a"].Wins != 0 {
		t.Errorf("Expected model-a wins to be 0, got %d", ratings["model-a"].Wins)
	}
	if ratings["model-b"].Losses != 0 {
		t.Errorf("Expected model-b losses to be 0, got %d", ratings["model-b"].Losses)
	}
}
