package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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

	category, confidence, _, err := classifier.ClassifyCategoryWithEntropy("unrelated request")
	if err == nil || err.Error() != "no category classification method available" {
		t.Fatalf("ClassifyCategoryWithEntropy error = %v, want no category classification method available", err)
	}
	if category != "" {
		t.Fatalf("ClassifyCategoryWithEntropy category = %q, want empty", category)
	}
	if confidence != 0 {
		t.Fatalf("ClassifyCategoryWithEntropy confidence = %v, want 0", confidence)
	}
}
