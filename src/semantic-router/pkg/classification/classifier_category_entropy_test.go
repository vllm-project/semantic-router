package classification

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
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

	classifier := &Classifier{
		Config:            &config.RouterConfig{},
		keywordClassifier: keywordClassifier,
	}

	_, _, _, err = classifier.ClassifyCategoryWithEntropyContext(context.Background(), "unrelated request")
	if err == nil || err.Error() != "no category classification method available" {
		t.Fatalf("unexpected error: %v", err)
	}
}

type contextCheckingMCP struct{ MockCategoryInference }

func (*contextCheckingMCP) ListCategories(context.Context) (*CategoryMapping, error) {
	return nil, nil
}

func (*contextCheckingMCP) ClassifyWithProbabilities(ctx context.Context, _ string) (tasks.ClassResultWithProbs, error) {
	return tasks.ClassResultWithProbs{}, ctx.Err()
}

func TestCategoryMCPFallbackRetainsCallerCancellation(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.TimeoutSeconds = 1
	classifier := &Classifier{Config: cfg, mcpCategoryInference: &contextCheckingMCP{}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := classifier.classifyMCPWithProbabilities(ctx, "canceled"); !errors.Is(err, context.Canceled) {
		t.Fatalf("MCP fallback lost caller context: %v", err)
	}
}
