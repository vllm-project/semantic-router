package routerreplay

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// scanStore is a database-backed store whose reads take scanTakes, or end with
// their context.
type scanStore struct {
	store.Storage
	scanTakes   time.Duration
	filters     store.QueryFilters
	optionReads int
}

func (s *scanStore) QueryPage(ctx context.Context, _ store.QueryFilters, _, _ int, _ bool) (store.RecordPage, error) {
	return store.RecordPage{}, s.wait(ctx)
}

func (s *scanStore) ScanMetadata(ctx context.Context, filters store.QueryFilters, visit func(store.Record) error) error {
	s.filters = filters
	if err := s.wait(ctx); err != nil {
		return err
	}
	return visit(store.Record{ID: "scanned"})
}

func (s *scanStore) ScanFilterOptions(_ context.Context, visit func(store.Record) error) error {
	s.optionReads++
	return visit(store.Record{Recipe: "evaluation"})
}

func (s *scanStore) wait(ctx context.Context) error {
	select {
	case <-time.After(s.scanTakes):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func TestRecorderReadsOutlastTheWriteBound(t *testing.T) {
	recorder := NewRecorder(&scanStore{Storage: store.NewMemoryStore(10, 0), scanTakes: 50 * time.Millisecond})
	recorder.operationTimeout = 10 * time.Millisecond

	visited := 0
	if err := recorder.ScanMetadata(context.Background(), store.QueryFilters{}, func(RoutingRecord) error {
		visited++
		return nil
	}); err != nil || visited != 1 {
		t.Fatalf("scan longer than the write bound: visited=%d err=%v", visited, err)
	}
	if _, _, err := recorder.QueryPage(context.Background(), store.QueryFilters{}, 1, 0, false); err != nil {
		t.Fatalf("page longer than the write bound: %v", err)
	}
}

func TestRecorderReadsEndWithTheReadBoundOrTheCaller(t *testing.T) {
	recorder := NewRecorder(&scanStore{Storage: store.NewMemoryStore(10, 0), scanTakes: time.Hour})
	recorder.readTimeout = 20 * time.Millisecond
	ignore := func(RoutingRecord) error { return nil }

	if err := recorder.ScanMetadata(context.Background(), store.QueryFilters{}, ignore); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("stalled scan error = %v, want the read bound", err)
	}

	recorder.readTimeout = time.Hour
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(20*time.Millisecond, cancel)
	if err := recorder.ScanMetadata(ctx, store.QueryFilters{}, ignore); !errors.Is(err, context.Canceled) {
		t.Fatalf("scan after its caller left: error = %v, want canceled", err)
	}
}

func TestRecorderReadsFilterOptionsOnlyForANarrowedScan(t *testing.T) {
	storage := &scanStore{Storage: store.NewMemoryStore(10, 0)}
	recorder := NewRecorder(storage)
	ignore := func(RoutingRecord) error { return nil }

	if separate, err := recorder.ScanFilterOptions(context.Background(), store.QueryFilters{}, ignore); separate || err != nil || storage.optionReads != 0 {
		t.Fatalf("unfiltered: separate=%v err=%v option reads=%d", separate, err, storage.optionReads)
	}
	filters := store.QueryFilters{Model: "model-a"}
	if separate, err := recorder.ScanFilterOptions(context.Background(), filters, ignore); !separate || err != nil || storage.optionReads != 1 {
		t.Fatalf("filtered: separate=%v err=%v option reads=%d", separate, err, storage.optionReads)
	}
	if err := recorder.ScanMetadata(context.Background(), filters, ignore); err != nil || storage.filters != filters {
		t.Fatalf("scan filters = %+v err=%v, want %+v", storage.filters, err, filters)
	}

	bounded := NewRecorder(store.NewMemoryStore(10, 0))
	if separate, err := bounded.ScanFilterOptions(context.Background(), filters, ignore); separate || err != nil {
		t.Fatalf("bounded store: separate=%v err=%v", separate, err)
	}
}
