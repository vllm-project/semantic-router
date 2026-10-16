package store

import (
	"cmp"
	"context"
	"errors"
	"slices"
	"strings"
)

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

// QueryReader provides paginated reads and a streaming metadata scan without
// loading the whole store. Implementations must not cap the scan.
type QueryReader interface {
	QueryPage(ctx context.Context, filters QueryFilters, limit, offset int, details bool) (RecordPage, error)
	ScanMetadata(ctx context.Context, visit func(Record) error) error
}

var errInvalidPageBounds = errors.New("replay query requires a positive limit and nonnegative offset")

func (f QueryFilters) empty() bool {
	return !f.RecipeSet && f.Recipe == "" && f.Decision == "" && f.Model == "" && f.SessionID == "" &&
		f.Search == "" && f.CacheStatus != "cached" && f.CacheStatus != "streamed"
}

// matches mirrors postgresReplayWhere.
func (f QueryFilters) matches(r Record) bool {
	if (f.RecipeSet || f.Recipe != "") && r.Recipe != f.Recipe {
		return false
	}
	if f.Decision != "" && r.Decision != f.Decision {
		return false
	}
	if f.SessionID != "" && r.SessionID != f.SessionID {
		return false
	}
	if f.Model != "" && r.SelectedModel != f.Model && r.OriginalModel != f.Model {
		return false
	}
	if f.Search != "" {
		search := strings.ToLower(f.Search)
		if !strings.Contains(strings.ToLower(r.RequestID), search) && !strings.Contains(strings.ToLower(r.Recipe), search) {
			return false
		}
	}
	switch f.CacheStatus {
	case "cached":
		return r.FromCache
	case "streamed":
		return r.Streaming
	}
	return true
}

// recordScan visits records newest first from position from until visit
// returns false.
type recordScan func(from int, visit func(Record) bool) error

// collectPage serves QueryPage from a scan. Unfiltered pages take the total
// from count and skip straight to the offset.
func collectPage(filters QueryFilters, limit, offset int, count func() (int, error), scan recordScan) (RecordPage, error) {
	if limit <= 0 || offset < 0 {
		return RecordPage{}, errInvalidPageBounds
	}
	page := RecordPage{Records: []Record{}}
	if filters.empty() {
		total, err := count()
		if err != nil {
			return RecordPage{}, err
		}
		page.Total, page.Offset = total, min(offset, total)
		err = scan(page.Offset, func(record Record) bool {
			page.Records = append(page.Records, record)
			return len(page.Records) < limit
		})
		return page, err
	}
	matched := 0
	err := scan(0, func(record Record) bool {
		if !filters.matches(record) {
			return true
		}
		if matched >= offset && len(page.Records) < limit {
			page.Records = append(page.Records, record)
		}
		matched++
		return true
	})
	page.Total, page.Offset = matched, min(offset, matched)
	return page, err
}

func scanAll(scan recordScan, visit func(Record) error) error {
	var visitErr error
	err := scan(0, func(record Record) bool {
		visitErr = visit(record)
		return visitErr == nil
	})
	return errors.Join(err, visitErr)
}

func listAll(scan recordScan) ([]Record, error) {
	records := []Record{}
	err := scan(0, func(record Record) bool {
		records = append(records, record)
		return true
	})
	return records, err
}

// newestFirst is the order every store pages in.
func newestFirst(a, b Record) int {
	if c := b.Timestamp.Compare(a.Timestamp); c != 0 {
		return c
	}
	return cmp.Compare(b.ID, a.ID)
}

// visitGroup sorts records into page order and visits them from position from.
func visitGroup(records []Record, from int, visit func(Record) bool) bool {
	slices.SortFunc(records, newestFirst)
	for _, record := range records[min(from, len(records)):] {
		if !visit(record) {
			return false
		}
	}
	return true
}
