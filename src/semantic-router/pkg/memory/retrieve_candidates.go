package memory

const (
	// Milvus accepts a vector-search window this large, with ef strictly above k.
	maxProjectScopedSearchTopK = 16_384
	// Valkey rejects HNSW EF_RUNTIME values above this supported maximum.
	maxValkeyVectorEfRuntime = 4_096
)

func retrieveSearchTopK(limit int, hybridSearch bool) int {
	searchTopK := limit * 4
	if hybridSearch {
		searchTopK = limit * 8
	}
	if searchTopK < 20 {
		searchTopK = 20
	}
	return searchTopK
}

// nextProjectScopedSearchTopK widens a vector query only when legacy rows can
// still be occupying the candidate window. Before project_id was persisted
// faithfully, an unscoped memory was indexed as project_id=default. A query
// for the explicit "default" project therefore needs to inspect past those
// legacy candidates before the metadata check can remove them.
func nextProjectScopedSearchTopK(projectID string, requestedTopK, returnedCount, projectMatches, maxTopK int) int {
	if projectID == "" || returnedCount < requestedTopK || projectMatches >= requestedTopK || requestedTopK >= maxTopK {
		return 0
	}

	next := requestedTopK * 2
	if next > maxTopK {
		next = maxTopK
	}
	if next <= requestedTopK {
		return 0
	}
	return next
}
