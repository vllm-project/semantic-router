package selection

import (
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func categoryWithScores(category string, scores map[string]float64) config.Category {
	c := config.Category{CategoryMetadata: config.CategoryMetadata{Name: category}}
	for model, score := range scores {
		c.ModelScores = append(c.ModelScores, config.ModelScore{Model: model, Score: score})
	}
	return c
}

func selectorWithPreloadedStorage(t *testing.T, ratings map[string]map[string]*ModelRating, storagePath string) *EloSelector {
	t.Helper()
	storage, err := NewFileEloStorage(storagePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := storage.SaveAllRatings(ratings); err != nil {
		t.Fatal(err)
	}
	e := NewEloSelector(&EloConfig{
		InitialRating:    DefaultEloRating,
		KFactor:          EloKFactor,
		CategoryWeighted: true,
		StoragePath:      storagePath,
	})
	if err := e.loadFromStorage(); err != nil {
		t.Fatal(err)
	}
	for cat, models := range ratings {
		for model, want := range models {
			if cat == "_global" {
				if got := e.globalRatings[model]; got == nil || got.Rating != want.Rating {
					t.Fatalf("precondition: restored global %s = %#v, want %v", model, got, want)
				}
				continue
			}
			if got := e.categoryRatings[cat][model]; got == nil || got.Rating != want.Rating {
				t.Fatalf("precondition: restored %s/%s = %#v, want %v", cat, model, got, want)
			}
		}
	}
	return e
}

func TestInitializeFromConfigPreservesRestoredCategoryRating(t *testing.T) {
	e := selectorWithPreloadedStorage(t,
		map[string]map[string]*ModelRating{
			"dev": {"model-a": {Model: "model-a", Rating: 1740}},
		},
		filepath.Join(t.TempDir(), "elo.json"),
	)

	e.InitializeFromConfig(
		map[string]config.ModelParams{},
		[]config.Category{categoryWithScores("dev", map[string]float64{"model-a": 0.7})},
	)

	if rating := e.categoryRatings["dev"]["model-a"]; rating.Rating != 1740 {
		t.Fatalf("restored category rating = %.1f, want the learned 1740", rating.Rating)
	}
}

func TestInitializeFromConfigSeedsMissingCategoryRating(t *testing.T) {
	e := selectorWithPreloadedStorage(t,
		map[string]map[string]*ModelRating{
			"dev": {"model-a": {Model: "model-a", Rating: 1740}},
		},
		filepath.Join(t.TempDir(), "elo.json"),
	)

	e.InitializeFromConfig(
		map[string]config.ModelParams{},
		[]config.Category{categoryWithScores("dev", map[string]float64{
			"model-a": 0.7,
			"model-b": 0.6,
		})},
	)

	if rating := e.categoryRatings["dev"]["model-b"]; rating.Rating != EloMinRatingFromScore+(0.6*EloRatingRange) {
		t.Fatalf("missing model-b rating = %.1f, want the static seed", rating.Rating)
	}
	if rating := e.categoryRatings["dev"]["model-a"]; rating.Rating != 1740 {
		t.Fatalf("restored model-a rating = %.1f, want the learned 1740", rating.Rating)
	}
}

func TestInitializeFromConfigPreservesRestoredGlobalRating(t *testing.T) {
	e := selectorWithPreloadedStorage(t,
		map[string]map[string]*ModelRating{
			"_global": {"model-g": {Model: "model-g", Rating: 1810}},
		},
		filepath.Join(t.TempDir(), "elo.json"),
	)

	e.InitializeFromConfig(
		map[string]config.ModelParams{"model-g": {}},
		[]config.Category{categoryWithScores("dev", map[string]float64{"model-g": 0.5})},
	)

	if rating := e.globalRatings["model-g"]; rating.Rating != 1810 {
		t.Fatalf("restored global rating = %.1f, want the learned 1810", rating.Rating)
	}
}

func TestStorageRoundTripKeepsLearnedRatingThroughInit(t *testing.T) {
	storagePath := filepath.Join(t.TempDir(), "elo.json")

	first := NewEloSelector(&EloConfig{
		InitialRating:    DefaultEloRating,
		KFactor:          EloKFactor,
		CategoryWeighted: true,
		StoragePath:      storagePath,
	})
	storage, err := NewFileEloStorage(storagePath)
	if err != nil {
		t.Fatal(err)
	}
	first.SetStorage(storage)
	if err := storage.SaveAllRatings(map[string]map[string]*ModelRating{
		"dev": {"model-a": {Model: "model-a", Rating: 1740}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := first.loadFromStorage(); err != nil {
		t.Fatal(err)
	}

	second := selectorWithPreloadedStorage(t,
		map[string]map[string]*ModelRating{
			"dev": {"model-a": {Model: "model-a", Rating: 1740}},
		},
		storagePath,
	)
	second.InitializeFromConfig(
		map[string]config.ModelParams{},
		[]config.Category{categoryWithScores("dev", map[string]float64{"model-a": 0.7})},
	)

	if rating := second.categoryRatings["dev"]["model-a"]; rating.Rating != 1740 {
		t.Fatalf("rating after reload and init = %.1f, want the learned 1740", rating.Rating)
	}
}
