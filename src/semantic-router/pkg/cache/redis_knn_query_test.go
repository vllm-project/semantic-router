//go:build !windows && cgo

package cache

import (
	"strings"
	"testing"
)

func TestPartitionedKNNQuery(t *testing.T) {
	tests := []struct {
		name        string
		model       string
		topK        int
		vectorField string
		want        string
	}{
		{
			name:        "default model partition",
			model:       "model-a",
			topK:        1,
			vectorField: "embedding",
			want:        "(@model:{model\\-a})=>[KNN 1 @embedding $vec AS vector_distance]",
		},
		{
			name:        "recipe namespaced partition",
			model:       "privacy::model/a",
			topK:        5,
			vectorField: "vec_field",
			want:        "(@model:{privacy\\:\\:model\\/a})=>[KNN 5 @vec_field $vec AS vector_distance]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := partitionedKNNQuery(tt.model, tt.topK, tt.vectorField); got != tt.want {
				t.Fatalf("partitionedKNNQuery() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestPartitionedKNNQueryCannotDropModelFilter(t *testing.T) {
	query := partitionedKNNQuery("model-a", 1, "embedding")
	if !strings.HasPrefix(query, "(@model:{") {
		t.Fatalf("KNN query must start with an exact model filter, got %q", query)
	}
	if strings.HasPrefix(query, "*=>") {
		t.Fatalf("unpartitioned vector query would permit cross-model hits: %q", query)
	}
}
