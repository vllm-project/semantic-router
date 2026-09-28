package classification

import (
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The anchors are close in text space (cosine .95), but are not substitutes
// for the image/audio query. Compression must be an explicit rule policy.
func TestEmbeddingCrossModalAnchorsArePreservedByDefault(t *testing.T) {
	for _, modality := range []config.QueryModality{config.QueryModalityImage, config.QueryModalityAudio} {
		t.Run(string(modality), func(t *testing.T) {
			for _, explicitCompression := range []bool{false, true} {
				rule := config.EmbeddingRule{Name: "media", QueryModality: modality, Candidates: []string{"anchor_a", "anchor_b"}, AggregationMethodConfiged: config.AggregationMethodMax}
				if explicitCompression {
					rule.PrototypeScoring = &config.PrototypeScoringConfig{Enabled: &explicitCompression}
				}
				classifier, err := NewEmbeddingClassifier([]config.EmbeddingRule{rule}, config.HNSWConfig{})
				if err != nil {
					t.Fatal(err)
				}
				classifier.candidateEmbeddings = map[string][]float32{
					"anchor_a": {1, 0}, "anchor_b": {0.95, float32(math.Sqrt(1 - 0.95*0.95))},
				}
				classifier.rebuildRulePrototypeBanks()
				scores, err := classifier.scoreRulesSlice([]float32{0, 1}, []config.EmbeddingRule{rule})
				if err != nil || len(scores) != 1 {
					t.Fatalf("scores: %+v, error: %v", scores, err)
				}
				if explicitCompression {
					if scores[0].PrototypeCount != 1 {
						t.Fatal("explicit compression must still be honored")
					}
				} else if scores[0].PrototypeCount != 2 || math.Abs(scores[0].Score-math.Sqrt(1-0.95*0.95)) > 1e-6 {
					t.Fatalf("cross-modal max lost the matching anchor: %+v", scores[0])
				}
			}
		})
	}
}
