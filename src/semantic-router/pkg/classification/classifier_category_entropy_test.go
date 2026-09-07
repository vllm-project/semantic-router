package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestClassifyCategoryWithEntropySkipsNilEmbeddingClassifier guards against a
// Go typed-nil bug: a nil *EmbeddingClassifier placed directly into a
// []interface{Classify(...)} literal becomes a non-nil interface value, so a
// nil check against the interface never trips and Classify is called on a
// nil receiver. With only the keyword classifier configured and no keyword
// match, this must fall through to the "no category classification method
// available" error rather than panic.
func TestClassifyCategoryWithEntropySkipsNilEmbeddingClassifier(t *testing.T) {
	keywordClassifier, err := NewKeywordClassifier([]config.KeywordRule{
		{
			Name:     "technical",
			Operator: "OR",
			Method:   "regex",
			Keywords: []string{"kubernetes"},
		},
	})
	if err != nil {
		t.Fatalf("NewKeywordClassifier: %v", err)
	}
	defer keywordClassifier.Free()

	classifier := &Classifier{
		Config:            &config.RouterConfig{},
		keywordClassifier: keywordClassifier,
	}

	_, _, _, err = classifier.ClassifyCategoryWithEntropy("unrelated request")
	if err == nil || err.Error() != "no category classification method available" {
		t.Fatalf("unexpected error: %v", err)
	}
}
