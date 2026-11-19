package store

import (
	"context"
	"strings"
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
	// Harness keeps the records of the harnesses it lists, comma-separated,
	// by FilterHarness. A string rather than a slice keeps the filters
	// comparable.
	Harness string
	// Since keeps the records at or after an instant; zero keeps every one.
	Since time.Time
	// RecipeSet distinguishes an explicit legacy empty recipe from no filter.
	RecipeSet bool
}

// HarnessListed reports whether a comma-separated harness list holds harness.
func HarnessListed(list, harness string) bool {
	for listed := range strings.SplitSeq(list, ",") {
		if listed == harness {
			return true
		}
	}
	return false
}

type RecordPage struct {
	Records []Record
	Total   int
	Offset  int
	// HarnessCounts counts the records every filter but Harness selects, by
	// FilterHarness.
	HarnessCounts map[string]int
}

// RunReader optionally reads one agent run's records, oldest first, without
// their captured bodies or prompt and with each tool trace's steps but not
// their payloads: what the run view needs of a session of thousands of calls.
type RunReader interface {
	QueryRun(ctx context.Context, filters QueryFilters) ([]Record, error)
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
