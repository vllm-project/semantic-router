package selection

import (
	"context"
	"testing"
)

// A repeated prompt_cache_key keeps a weighted tie on the model that served
// the key; a request without the key keeps the tie on the first candidate.
func TestMultiFactorTieFollowsThePromptCacheKey(t *testing.T) {
	s := buildMFSelector(&MultiFactorConfig{Weights: MultiFactorWeights{Quality: 0.2, Cost: 0.1}}, nil, nil, nil, nil)

	withKey := makeAffCtx(0, 0, 500, "b", "", nil)
	withKey.PromptCacheKey = "conversation-1"
	result, err := s.Select(context.Background(), &SelectionContext{CandidateModels: candidates("a", "b"), CacheAffinityCtx: withKey})
	if err != nil || result.SelectedModel != "b" {
		t.Fatalf("with the key: selected %+v, error %v; want b, the model that served it", result, err)
	}

	result, err = s.Select(context.Background(), &SelectionContext{CandidateModels: candidates("a", "b"), CacheAffinityCtx: makeAffCtx(0, 0, 500, "", "", nil)})
	if err != nil || result.SelectedModel != "a" {
		t.Fatalf("without the key: selected %+v, error %v; want a", result, err)
	}
}
