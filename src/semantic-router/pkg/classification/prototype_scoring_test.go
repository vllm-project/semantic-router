package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestPrototypeBankClustersAndSelectsMedoids(t *testing.T) {
	enabled := true
	bank := newPrototypeBank([]prototypeExample{
		{Key: "alpha-left", Text: "alpha-left", Embedding: makeEmbedding(1.0, 0.0)},
		{Key: "alpha-center", Text: "alpha-center", Embedding: makeEmbedding(0.99, 0.08)},
		{Key: "alpha-right", Text: "alpha-right", Embedding: makeEmbedding(0.97, 0.16)},
		{Key: "beta-left", Text: "beta-left", Embedding: makeEmbedding(0.0, 1.0)},
		{Key: "beta-center", Text: "beta-center", Embedding: makeEmbedding(0.08, 0.99)},
		{Key: "", Text: "beta-center", Embedding: makeEmbedding(0.08, 0.99)},
	}, config.PrototypeScoringConfig{
		Enabled:                    &enabled,
		ClusterSimilarityThreshold: 0.96,
		MaxPrototypes:              8,
	})

	if bank == nil {
		t.Fatal("expected prototype bank")
	}
	if len(bank.prototypes) != 2 {
		t.Fatalf("expected 2 prototypes after clustering and dedupe, got %+v", bank.prototypes)
	}
	if bank.prototypes[0].Text != "alpha-center" {
		t.Fatalf("expected alpha cluster medoid to be alpha-center, got %+v", bank.prototypes[0])
	}
	if bank.prototypes[1].Text != "beta-center" {
		t.Fatalf("expected beta cluster medoid to be beta-center, got %+v", bank.prototypes[1])
	}
	if bank.prototypes[0].ClusterSize != 3 {
		t.Fatalf("expected alpha cluster size 3, got %+v", bank.prototypes[0])
	}
	if bank.prototypes[1].ClusterSize != 2 {
		t.Fatalf("expected beta cluster size 2 after dedupe, got %+v", bank.prototypes[1])
	}
}
