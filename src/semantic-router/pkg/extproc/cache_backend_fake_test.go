package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"

// similarEntries is a fake backend's single lookup result as the entries
// LookupSimilar returns: a found result is the one hit, for the cached
// request query and at similarity 1 unless it names its own; a miss carries
// its similarity.
func similarEntries(result cache.LookupResult, query string) cache.SimilarEntries {
	if !result.Found {
		return cache.SimilarEntries{Closest: result.Similarity}
	}
	if result.Query == "" {
		result.Query = query
	}
	if result.Similarity == 0 {
		result.Similarity = 1
	}
	return cache.SimilarEntries{Hits: []cache.LookupResult{result}}
}
