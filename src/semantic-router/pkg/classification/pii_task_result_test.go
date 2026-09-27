package classification

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type rawPIITask struct {
	result tasks.TokenClassificationResult
	err    error
}

func (r rawPIITask) ClassifyTokens(context.Context, string) (tasks.TokenClassificationResult, error) {
	return r.result, r.err
}

func TestOwnedPIIOutsideFilteringPreservesPartialMetadata(t *testing.T) {
	available := true
	boundary := 9
	raw := tasks.TokenClassificationResult{ScoresAvailable: &available, TruncatedAt: &boundary, Entities: []tasks.TokenEntity{{EntityType: "class_0", Text: "中文", Start: 0, End: 6, Confidence: .99}, {EntityType: "PERSON", Text: "人", Start: 6, End: 9, Confidence: .8}}}
	c := &Classifier{PIIMapping: &PIIMapping{IdxToLabel: map[string]string{"0": "NONE", "1": "PERSON"}}, piiInference: rawPIITask{raw, tasks.ErrTokenSpansTruncated}}
	result, err := c.classifyPIITokens(context.Background(), "中文人")
	if !errors.Is(err, tasks.ErrTokenSpansTruncated) || result.TruncatedAt != &boundary {
		t.Fatalf("partial metadata changed: %v %v", result, err)
	}
	if len(result.Entities) != 1 || result.Entities[0].EntityType != "PERSON" || result.Entities[0].Start != 6 {
		t.Fatalf("wrong entity spans: %v", result.Entities)
	}
	if len(raw.Entities) != 2 {
		t.Fatal("consumer mutated provider output")
	}
}

func TestOwnedPIIRequiresDeclaredProbabilities(t *testing.T) {
	unavailable := false
	for _, flag := range []*bool{nil, &unavailable} {
		c := &Classifier{piiInference: rawPIITask{result: tasks.TokenClassificationResult{ScoresAvailable: flag}}}
		if _, err := c.classifyPIITokens(context.Background(), "text"); !errors.Is(err, tasks.ErrProbabilitiesUnavailable) {
			t.Fatalf("unscored input interpreted as safe: %v", err)
		}
	}
}
