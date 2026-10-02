package modelselection

import (
	"reflect"
	"testing"
)

func TestCategoryFeatureContract(t *testing.T) {
	// This order is part of the Python-exported model artifact format.
	categories := []string{"math", "physics", "chemistry", "biology", "computer science", "history", "economics", "business", "law", "health", "psychology", "philosophy", "other", "unknown"}
	if !reflect.DeepEqual(VSRCategories, categories) {
		t.Fatalf("artifact category order changed: %v", VSRCategories)
	}
	for index, category := range categories {
		t.Run(category, func(t *testing.T) {
			embedding := []float64{0.25, 0.75}
			want := make([]float64, 16)
			copy(want, embedding)
			want[len(embedding)+index] = 1
			got := CombineEmbeddingWithCategory(embedding, category)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("features = %v, want %v", got, want)
			}
			got[0] = 99
			if embedding[0] != 0.25 {
				t.Fatal("feature vector aliases the caller's embedding")
			}
		})
	}
	for _, category := range []string{"", "unrecognized-domain"} {
		if got, want := CategoryToOneHot(category), CategoryToOneHot("other"); !reflect.DeepEqual(got, want) {
			t.Fatalf("unrecognized category %q = %v, want other %v", category, got, want)
		}
	}
}
