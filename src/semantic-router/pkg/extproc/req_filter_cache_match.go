package extproc

import (
	"context"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
)

// responseCacheIndex is the semantic tier's match.Index: it recalls the
// cached requests of one request's cache partition, and keeps the entries it
// recalled so the lookup serves the one the match keeps first.
type responseCacheIndex struct {
	service  *cache.ResponseCacheService
	identity cache.CacheIdentity
	maxAge   *time.Duration
	recalled []cache.CacheResult
	closest  float32
}

// Recall returns the fresh cached requests most similar to query, each by its
// position among the recalled entries, with the request it cached as the text
// a same_request judge reads.
func (i *responseCacheIndex) Recall(ctx context.Context, query string, recall config.MatchRecall) ([]match.Scored, error) {
	identity := i.identity
	identity.SemanticQuery = query
	bounds := cache.SimilarityBounds{MinSimilarity: -1, TopK: recall.TopK}
	if recall.MinSimilarity != nil {
		bounds.MinSimilarity = float32(*recall.MinSimilarity)
	}
	result, err := i.service.LookupSemantic(ctx, cache.SemanticLookup{Identity: identity, Bounds: bounds, MaxAge: i.maxAge})
	if err != nil {
		return nil, err
	}
	i.recalled, i.closest = result.Candidates, result.Closest
	scored := make([]match.Scored, len(result.Candidates))
	for index, candidate := range result.Candidates {
		scored[index] = match.Scored{
			Candidate:  match.Candidate{ID: strconv.Itoa(index), Text: candidate.Query},
			Similarity: float64(candidate.Similarity),
		}
	}
	return scored, nil
}

// served is the recalled entry the match kept first; a miss carries the
// closest similarity recall saw.
func (i *responseCacheIndex) served(result match.Result, err error) cache.CacheResult {
	if err != nil || len(result.Kept) == 0 {
		return cache.CacheResult{Similarity: i.closest}
	}
	index, convErr := strconv.Atoi(result.Kept[0].ID)
	if convErr != nil || index < 0 || index >= len(i.recalled) {
		return cache.CacheResult{Similarity: i.closest}
	}
	return i.recalled[index]
}

// judgeStage is the judge stage a match ran, if any.
func judgeStage(result match.Result) (match.Stage, bool) {
	for _, stage := range result.Stages {
		if stage.Name == match.StageJudge {
			return stage, true
		}
	}
	return match.Stage{}, false
}
