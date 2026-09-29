package k8s

import (
	"testing"

	v1alpha1 "github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
)

func TestConvertEmbeddingCandidateBanks(t *testing.T) {
	input := v1alpha1.Signals{Embeddings: []v1alpha1.EmbeddingSignal{{Name: "mixed", Candidates: []string{"positive"}, ImageCandidates: []string{"./positive.png"}, NegativeCandidates: []string{"negative"}, NegativeImageCandidates: []string{"./negative.png"}}}}
	converted := convertSignals(input)
	got := converted.Embeddings[0]
	if len(got.Candidates) != 1 || len(got.ImageCandidates) != 1 || len(got.NegativeCandidates) != 1 || len(got.NegativeImageCandidates) != 1 {
		t.Fatalf("lost candidate banks: %+v", got)
	}
	clone := input.DeepCopy()
	clone.Embeddings[0].NegativeImageCandidates[0] = "changed"
	if input.Embeddings[0].NegativeImageCandidates[0] != "./negative.png" {
		t.Fatal("deep copy shares candidate arrays")
	}
}
