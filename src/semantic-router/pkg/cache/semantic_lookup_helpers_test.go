package cache

// atThreshold bounds a lookup to its most similar entry at threshold.
func atThreshold(threshold float32) SimilarityBounds {
	return SimilarityBounds{MinSimilarity: threshold, TopK: 1}
}

// semanticHit is a recall's most similar candidate, or a miss that carries
// the closest similarity the lookup saw.
func semanticHit(recall SemanticRecall, err error) (CacheResult, error) {
	if err != nil {
		return CacheResult{}, err
	}
	if len(recall.Candidates) == 0 {
		return CacheResult{HitKind: HitKindMiss, Similarity: recall.Closest}, nil
	}
	return recall.Candidates[0], nil
}
