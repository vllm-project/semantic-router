package store

import (
	"cmp"
	"context"
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"time"
)

type openContractStore func(t *testing.T, ttlSeconds int) Storage

// runStoreContract checks the paging, retention and size behavior every
// replay backend shares.
func runStoreContract(t *testing.T, open openContractStore, records int) {
	t.Run("pages", func(t *testing.T) { checkContractPages(t, open(t, 0), records) })
	t.Run("ttl", func(t *testing.T) { checkContractTTL(t, open(t, 60)) })
	t.Run("size", func(t *testing.T) { checkContractSize(t, open(t, 0)) })
}

func checkContractPages(t *testing.T, s Storage, records int) {
	ctx := context.Background()
	base := time.Now().UTC().Add(-time.Hour).Truncate(time.Second)
	added := make([]Record, 0, records)
	// Insert out of order, so stores must sort rather than rely on arrival.
	for _, i := range rand.New(rand.NewPCG(1, 2)).Perm(records) {
		record := Record{
			ID:            fmt.Sprintf("rec-%05d", i),
			RequestID:     fmt.Sprintf("Req-%05d", i),
			Decision:      []string{"a", "b"}[i%2],
			SelectedModel: "model",
			// Three records per second, so ties are ordered by ID.
			Timestamp: base.Add(time.Duration(i/3) * time.Second),
		}
		if _, err := s.Add(ctx, record); err != nil {
			t.Fatalf("add %s: %v", record.ID, err)
		}
		added = append(added, record)
	}
	slices.SortFunc(added, func(a, b Record) int {
		if c := b.Timestamp.Compare(a.Timestamp); c != 0 {
			return c
		}
		return cmp.Compare(b.ID, a.ID)
	})

	for _, tc := range []struct {
		filters QueryFilters
		limit   int
	}{
		{QueryFilters{}, 7},
		{QueryFilters{Decision: "a", Search: "REQ-000"}, 5},
	} {
		var want, got []string
		for _, record := range added {
			if tc.filters.matches(record) {
				want = append(want, record.ID)
			}
		}
		for offset := 0; ; offset += tc.limit {
			page, err := s.QueryPage(ctx, tc.filters, tc.limit, offset, true)
			if err != nil {
				t.Fatalf("page at %d: %v", offset, err)
			}
			if page.Total != len(want) {
				t.Fatalf("filters %+v: total %d, want %d", tc.filters, page.Total, len(want))
			}
			for _, record := range page.Records {
				got = append(got, record.ID)
			}
			if len(page.Records) < tc.limit {
				break
			}
		}
		if !slices.Equal(got, want) {
			t.Fatalf("filters %+v: paged %d records, want %d in timestamp, ID order", tc.filters, len(got), len(want))
		}
	}

	scanned := 0
	if err := s.ScanMetadata(ctx, func(Record) error { scanned++; return nil }); err != nil {
		t.Fatal(err)
	}
	if scanned != records {
		t.Fatalf("scanned %d records, want %d", scanned, records)
	}
}

func checkContractTTL(t *testing.T, s Storage) {
	ctx := context.Background()
	now := time.Now().UTC()
	for _, record := range []Record{
		{ID: "expired", Timestamp: now.Add(-2 * time.Hour)},
		{ID: "live", Timestamp: now},
	} {
		if _, err := s.Add(ctx, record); err != nil {
			t.Fatal(err)
		}
	}
	page, err := s.QueryPage(ctx, QueryFilters{}, 10, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if page.Total != 1 || len(page.Records) != 1 || page.Records[0].ID != "live" {
		t.Fatalf("page has total %d and %d records, want only the live record", page.Total, len(page.Records))
	}
	if _, found, err := s.Get(ctx, "expired"); err != nil || found {
		t.Fatalf("expired record found=%v err=%v", found, err)
	}
	scanned := 0
	if err := s.ScanMetadata(ctx, func(Record) error { scanned++; return nil }); err != nil || scanned != 1 {
		t.Fatalf("scanned %d records (err %v), want 1", scanned, err)
	}
}

func checkContractSize(t *testing.T, s Storage) {
	ctx := context.Background()
	trace := func(raw string) *ToolTrace {
		return &ToolTrace{Steps: []ToolTraceStep{{Type: "tool_result", RawOutput: raw, Output: "output"}}}
	}
	now := time.Now().UTC()
	for _, record := range []Record{
		{ID: "small", Timestamp: now, ToolTrace: trace("raw")},
		{ID: "large", Timestamp: now, RequestBody: "request", ToolTrace: trace(strings.Repeat("r", 2*MaxRecordBytes))},
	} {
		if _, err := s.Add(ctx, record); err != nil {
			t.Fatal(err)
		}
	}

	small, found, err := s.Get(ctx, "small")
	if err != nil || !found {
		t.Fatalf("small record found=%v err=%v", found, err)
	}
	if step := small.ToolTrace.Steps[0]; step.RawOutput != "raw" || step.Truncated {
		t.Fatalf("small record raw output was cut: %+v", step)
	}

	large, found, err := s.Get(ctx, "large")
	if err != nil || !found {
		t.Fatalf("large record found=%v err=%v", found, err)
	}
	step := large.ToolTrace.Steps[0]
	if len(step.RawOutput) >= MaxRecordBytes || !step.Truncated || step.Output != "output" || large.RequestBody != "request" {
		t.Fatalf("large record kept %d raw bytes, truncated=%v, output=%q, request=%q", len(step.RawOutput), step.Truncated, step.Output, large.RequestBody)
	}

	// Each '<' serializes to six bytes, so cutting the raw body by the JSON
	// excess removes all of it before the record fits.
	if _, err := s.Add(ctx, Record{ID: "escaped", Timestamp: now, RequestBody: strings.Repeat("<", MaxRecordBytes)}); err != nil {
		t.Fatalf("escaped record: %v", err)
	}
	escaped, found, err := s.Get(ctx, "escaped")
	if err != nil || !found || !escaped.RequestBodyTruncated {
		t.Fatalf("escaped record found=%v err=%v truncated=%v", found, err, escaped.RequestBodyTruncated)
	}
}

func TestMemoryStoreContract(t *testing.T) {
	runStoreContract(t, func(t *testing.T, ttlSeconds int) Storage {
		return NewMemoryStore(5000, ttlSeconds)
	}, 2500)
}

func TestCollectPageStopsAfterThePage(t *testing.T) {
	visited := 0
	scan := func(from int, visit func(Record) bool) error {
		for i := from; i < 1000; i++ {
			visited++
			if !visit(Record{ID: fmt.Sprint(i)}) {
				return nil
			}
		}
		return nil
	}
	count := func() (int, error) { return 1000, nil }
	page, err := collectPage(QueryFilters{}, 10, 500, count, scan)
	if err != nil {
		t.Fatal(err)
	}
	if visited != 10 || page.Total != 1000 || page.Records[0].ID != "500" {
		t.Fatalf("visited %d records for a 10-record page at offset 500", visited)
	}
}
