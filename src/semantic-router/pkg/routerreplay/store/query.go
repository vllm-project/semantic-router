package store

import "context"

// QueryFilters is the metadata-only filter contract shared by replay lists and
// storage queries. Search is a literal, case-insensitive request-ID/recipe match.
type QueryFilters struct {
	Search, Recipe, Decision, Model, CacheStatus, SessionID string
	// RecipeSet distinguishes an explicit legacy empty recipe from no filter.
	RecipeSet bool
}

type RecordPage struct {
	Records []Record
	Total   int
	Offset  int
}

// QueryReader optionally provides database-side pagination and a streaming
// metadata scan. Implementations must not cap the scan or read captured bodies.
// Reader remains the compatibility contract for bounded in-memory stores.
type QueryReader interface {
	QueryPage(ctx context.Context, filters QueryFilters, limit, offset int, details bool) (RecordPage, error)
	ScanMetadata(ctx context.Context, visit func(Record) error) error
}
