package config

// ResponseCacheDefaultSimilarity is the cache backend's default similarity
// bound: the response cache's default recall bound. Each lookup passes the
// bound of the match block its request runs.
func ResponseCacheDefaultSimilarity() float32 {
	return float32(*MatchResponseCache.Default().Recall.MinSimilarity)
}
