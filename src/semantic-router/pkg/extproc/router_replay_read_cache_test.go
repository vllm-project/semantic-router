package extproc

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func (c *replayReadCache[V]) waitersFor(key string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if flight := c.flights[key]; flight != nil {
		return flight.waiters
	}
	return 0
}

func waitForReplayRead(t *testing.T, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !condition() {
		if time.Now().After(deadline) {
			t.Fatal("replay read did not reach the expected state")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestReplayReadCacheAnswersWithinItsTTL(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	cache := replayReadCache[int]{now: func() time.Time { return now }}
	scans := 0
	scan := func(context.Context) (int, error) {
		scans++
		return scans, nil
	}
	for _, step := range []struct {
		key     string
		advance time.Duration
		want    int
	}{
		{"model=a", 0, 1},
		{"model=a", replayReadCacheTTL - time.Second, 1},
		{"model=b", 0, 2},
		{"model=a", 2 * time.Second, 3},
	} {
		now = now.Add(step.advance)
		got, err := cache.get(context.Background(), step.key, scan)
		if err != nil || got != step.want {
			t.Fatalf("get(%q) %s later = %d, %v; want %d", step.key, step.advance, got, err, step.want)
		}
	}
}

func TestReplayReadCacheSharesOneScanAmongWaiters(t *testing.T) {
	var cache replayReadCache[string]
	release := make(chan struct{})
	var scans atomic.Int32
	scan := func(context.Context) (string, error) {
		scans.Add(1)
		<-release
		return "aggregate", nil
	}
	const callers = 8
	var wg sync.WaitGroup
	answers := make(chan string, callers)
	for range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			value, err := cache.get(context.Background(), "filters", scan)
			if err != nil {
				t.Error(err)
			}
			answers <- value
		}()
	}
	waitForReplayRead(t, func() bool { return cache.waitersFor("filters") == callers })
	close(release)
	wg.Wait()
	close(answers)
	for answer := range answers {
		if answer != "aggregate" {
			t.Fatalf("caller answer = %q", answer)
		}
	}
	if got := scans.Load(); got != 1 {
		t.Fatalf("scans for %d concurrent callers = %d, want 1", callers, got)
	}
}

func TestReplayReadCacheStopsTheScanOnceEveryCallerLeft(t *testing.T) {
	var cache replayReadCache[string]
	stopped := make(chan struct{})
	scan := func(ctx context.Context) (string, error) {
		<-ctx.Done()
		close(stopped)
		return "", ctx.Err()
	}
	first, leaveFirst := context.WithCancel(context.Background())
	second, leaveSecond := context.WithCancel(context.Background())
	results := make(chan error, 2)
	for _, ctx := range []context.Context{first, second} {
		go func() {
			_, err := cache.get(ctx, "filters", scan)
			results <- err
		}()
	}
	waitForReplayRead(t, func() bool { return cache.waitersFor("filters") == 2 })

	leaveFirst()
	if err := <-results; !errors.Is(err, context.Canceled) {
		t.Fatalf("first caller error = %v, want canceled", err)
	}
	select {
	case <-stopped:
		t.Fatal("scan stopped while a caller still waited for it")
	case <-time.After(20 * time.Millisecond):
	}
	leaveSecond()
	if err := <-results; !errors.Is(err, context.Canceled) {
		t.Fatalf("second caller error = %v, want canceled", err)
	}
	select {
	case <-stopped:
	case <-time.After(2 * time.Second):
		t.Fatal("scan ran on after every caller left")
	}

	value, err := cache.get(context.Background(), "filters", func(context.Context) (string, error) { return "fresh", nil })
	if err != nil || value != "fresh" {
		t.Fatalf("caller after the stopped scan = %q, %v; want a new scan", value, err)
	}
}

func TestReplayReadCacheKeepsNoFailure(t *testing.T) {
	var cache replayReadCache[int]
	unavailable := errors.New("database unavailable")
	if _, err := cache.get(context.Background(), "filters", func(context.Context) (int, error) { return 0, unavailable }); !errors.Is(err, unavailable) {
		t.Fatalf("failed scan error = %v", err)
	}
	if got, err := cache.get(context.Background(), "filters", func(context.Context) (int, error) { return 7, nil }); err != nil || got != 7 {
		t.Fatalf("scan after a failure = %d, %v; want a new scan", got, err)
	}
}

func TestReplayReadCacheBoundsItsEntries(t *testing.T) {
	var cache replayReadCache[int]
	for i := range 2 * replayReadCacheEntries {
		if _, err := cache.get(context.Background(), fmt.Sprint(i), func(context.Context) (int, error) { return i, nil }); err != nil {
			t.Fatal(err)
		}
	}
	if kept := len(cache.entries); kept > replayReadCacheEntries {
		t.Fatalf("cache keeps %d entries, want at most %d", kept, replayReadCacheEntries)
	}
}

// Two queries share a cached answer only when they select the same records, so
// the key names every filter, in the form the parser reads back.
func TestRouterReplayFilterCacheKeyNamesEveryFilter(t *testing.T) {
	filters, err := parseRouterReplayFilters(url.Values{
		"search": {"Request-1"}, "recipe": {"agents"}, "decision": {"code"}, "model": {"model-a"},
		"cache_status": {"cached"}, "session_id": {"session-1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	keyed, err := url.ParseQuery(filters.cacheKey())
	if err != nil {
		t.Fatal(err)
	}
	if again, err := parseRouterReplayFilters(keyed); err != nil || again != filters {
		t.Fatalf("cache key %q reads back as %+v, %v; want %+v", filters.cacheKey(), again, err, filters)
	}
	if reflect.TypeOf(filters).NumField() != 6 {
		t.Fatal("a new replay filter needs a place in cacheKey and in this test")
	}
	if (routerReplayFilters{}).cacheKey() != "" {
		t.Fatal("no filters must key as the whole history")
	}
}

// A reload builds a new router, whose cache starts empty; the retired router's
// cache goes with it.
func TestRouterReplayAggregateCacheBelongsToItsRouter(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	add := func(id string) {
		t.Helper()
		if _, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: id, Timestamp: time.Now(), LifecycleState: routerreplay.LifecycleCompleted}); err != nil {
			t.Fatal(err)
		}
	}
	recordCount := func(router *OpenAIRouter) float64 {
		t.Helper()
		response := router.handleRouterReplayAPI(context.Background(), "GET", routerReplayAggregatePath)
		return decodeJSONBody(t, response.GetImmediateResponse().Body)["record_count"].(float64)
	}

	add("first")
	generation := &OpenAIRouter{ReplayRecorder: recorder, ReplayStoreShared: true}
	if got := recordCount(generation); got != 1 {
		t.Fatalf("first aggregate = %v, want 1", got)
	}
	add("second")
	if got := recordCount(generation); got != 1 {
		t.Fatalf("aggregate within the cache TTL = %v, want the cached 1", got)
	}
	reloaded := &OpenAIRouter{ReplayRecorder: recorder, ReplayStoreShared: true}
	if got := recordCount(reloaded); got != 2 {
		t.Fatalf("reloaded router aggregate = %v, want its own scan of 2", got)
	}
}

// narrowingReplayStore filters its scan by model, as a database does, and
// counts the reads.
type narrowingReplayStore struct {
	*store.MemoryStore
	records     []store.Record
	scanned     []store.QueryFilters
	optionReads int
}

func (s *narrowingReplayStore) QueryPage(context.Context, store.QueryFilters, int, int, bool) (store.RecordPage, error) {
	return store.RecordPage{}, nil
}

func (s *narrowingReplayStore) ScanMetadata(_ context.Context, filters store.QueryFilters, visit func(store.Record) error) error {
	s.scanned = append(s.scanned, filters)
	for _, record := range s.records {
		if filters.Model == "" || record.SelectedModel == filters.Model {
			if err := visit(record); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *narrowingReplayStore) ScanFilterOptions(_ context.Context, visit func(store.Record) error) error {
	s.optionReads++
	for _, record := range s.records {
		if err := visit(store.Record{Recipe: record.Recipe, Decision: record.Decision, SelectedModel: record.SelectedModel}); err != nil {
			return err
		}
	}
	return nil
}

func TestRouterReplayFilteredAggregateNarrowsTheScanAndKeepsEveryOption(t *testing.T) {
	storage := &narrowingReplayStore{MemoryStore: store.NewMemoryStore(1, 0), records: []store.Record{
		{ID: "a", Recipe: "agents", Decision: "code", SelectedModel: "model-a", LifecycleState: routerreplay.LifecycleCompleted},
		{ID: "b", Recipe: "chat", Decision: "fast", SelectedModel: "model-b", LifecycleState: routerreplay.LifecycleCompleted},
	}}
	router := &OpenAIRouter{ReplayRecorder: routerreplay.NewRecorder(storage), ReplayStoreShared: true}
	aggregate := func(query string) map[string]interface{} {
		t.Helper()
		return decodeJSONBody(t, router.handleRouterReplayAPI(context.Background(), "GET", routerReplayAggregatePath+query).GetImmediateResponse().Body)
	}

	filtered := aggregate("?model=model-a")
	if filtered["record_count"] != float64(1) || storage.optionReads != 1 || storage.scanned[0].Model != "model-a" {
		t.Fatalf("filtered aggregate count=%v option reads=%d scan=%+v", filtered["record_count"], storage.optionReads, storage.scanned)
	}
	all := aggregate("")
	if all["record_count"] != float64(2) || storage.optionReads != 1 || storage.scanned[1] != (store.QueryFilters{}) {
		t.Fatalf("unfiltered aggregate count=%v option reads=%d scan=%+v", all["record_count"], storage.optionReads, storage.scanned)
	}
	for _, payload := range []map[string]interface{}{filtered, all} {
		if fmt.Sprint(payload["available_models"]) != "[model-a model-b]" || fmt.Sprint(payload["available_recipes"]) != "[agents chat]" {
			t.Fatalf("filter options = %v %v, want every record's", payload["available_models"], payload["available_recipes"])
		}
	}
}
