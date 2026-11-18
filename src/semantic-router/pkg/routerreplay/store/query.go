package store

import (
	"context"
	"time"
)

// QueryFilters is the metadata-only filter contract shared by replay lists and
// storage queries. Search is a literal, case-insensitive match on the request
// ID, the recipe or the session ID.
type QueryFilters struct {
	Search, Recipe, Decision, Model, CacheStatus, SessionID string
	// RequestKind and the review fields match exactly; a review's outcome and
	// risk exist only once it completed.
	RequestKind, ReviewStatus, ReviewOutcome, ReviewRisk string
	// TurnID matches a record's turn_id or parent_turn_id: the requests one
	// agent turn sent and the ones it started, such as its approval reviews.
	TurnID string
	// Since keeps the records at or after an instant; zero keeps every one.
	Since time.Time
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
	// ScanMetadata visits the records the filters select, in no set order.
	ScanMetadata(ctx context.Context, filters QueryFilters, visit func(Record) error) error
	// ScanFilterOptions visits one record for each distinct combination of the
	// fields the replay filters offer: recipe, decision, selected and original
	// model, and request kind. Its other fields stay empty.
	ScanFilterOptions(ctx context.Context, visit func(Record) error) error
}
