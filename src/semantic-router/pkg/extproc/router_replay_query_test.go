package extproc

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

type replayQueryTestStore struct {
	*store.MemoryStore
	page func(store.QueryFilters, int, int, bool) (store.RecordPage, error)
	scan func(func(store.Record) error) error
}

func (s *replayQueryTestStore) QueryPage(_ context.Context, f store.QueryFilters, l, o int, d bool) (store.RecordPage, error) {
	return s.page(f, l, o, d)
}

func (s *replayQueryTestStore) ScanMetadata(_ context.Context, v func(store.Record) error) error {
	return s.scan(v)
}

func TestRouterReplayDatabasePageIncludesOlderRecords(t *testing.T) {
	storage := &replayQueryTestStore{page: func(f store.QueryFilters, l, o int, d bool) (store.RecordPage, error) {
		if f.Recipe != "evaluation" || l != 5 || o != 12225 || d {
			t.Fatalf("query lost filters/page: %+v %d %d %v", f, l, o, d)
		}
		records := make([]store.Record, 5)
		for i := range records {
			records[i].ID = fmt.Sprintf("older-%d", i)
		}
		return store.RecordPage{Records: records, Total: 12230, Offset: o}, nil
	}}
	router := &OpenAIRouter{ReplayRecorder: routerreplay.NewRecorder(storage), ReplayStoreShared: true}
	response := router.handleRouterReplayAPI("GET", routerReplayAPIBasePath+"?recipe=evaluation&offset=12225&limit=5")
	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if body["total"] != float64(12230) || body["count"] != float64(5) || body["has_more"] != false {
		t.Fatalf("incorrect DB page: %#v", body)
	}
}

func TestRouterReplayStreamingAggregateMatchesAllRecords(t *testing.T) {
	records := make([]routerreplay.RoutingRecord, 12250)
	for i := range records {
		currency := "USD"
		if i%3 == 0 {
			currency = "EUR"
		}
		records[i] = routerreplay.RoutingRecord{RequestID: fmt.Sprintf("request-%d", i), Recipe: "evaluation", Decision: fmt.Sprintf("decision-%d", i%13), SelectedModel: fmt.Sprintf("model-%d", i%12), OriginalModel: "balance", LifecycleState: routerreplay.LifecycleCompleted, PromptTokens: replayIntPtr(10), CompletionTokens: replayIntPtr(2), TotalTokens: replayIntPtr(12), ActualCost: replayFloatPtr(.25), BaselineCost: replayFloatPtr(1), CostSavings: replayFloatPtr(.75), Currency: &currency, BaselineModel: replayStringPtr("base"), Signals: routerreplay.Signal{Keyword: []string{"a", "b"}}}
		if i%7 == 0 {
			records[i].LifecycleState = routerreplay.LifecycleFailed
		}
		if i%11 == 0 {
			records[i].TotalTokens = nil
		}
	}
	storage := &replayQueryTestStore{scan: func(visit func(store.Record) error) error {
		for _, record := range records {
			if err := visit(record); err != nil {
				return err
			}
		}
		return nil
	}}
	router := &OpenAIRouter{ReplayRecorder: routerreplay.NewRecorder(storage), ReplayStoreShared: true}
	for _, filters := range []routerReplayFilters{{}, {model: "model-1"}, {search: "request-122"}} {
		got, err := router.queryRouterReplayAggregate(filters)
		want := buildRouterReplayAggregatePayload(records, filterRouterReplayRecords(records, filters))
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("stream aggregate differs for %+v: got=%+v want=%+v err=%v", filters, got, want, err)
		}
	}
}

func TestRouterReplayQueryFailureDoesNotLookEmpty(t *testing.T) {
	unavailable := errors.New("database unavailable")
	storage := &replayQueryTestStore{page: func(store.QueryFilters, int, int, bool) (store.RecordPage, error) {
		return store.RecordPage{}, unavailable
	}, scan: func(func(store.Record) error) error { return unavailable }}
	router := &OpenAIRouter{ReplayRecorder: routerreplay.NewRecorder(storage), ReplayStoreShared: true}
	for _, path := range []string{routerReplayAPIBasePath, routerReplayAggregatePath, routerReplayTrajectoryPath + "?session_id=old-session"} {
		response := router.handleRouterReplayAPI("GET", path)
		if response.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_InternalServerError {
			t.Fatalf("query failure was hidden for %s", path)
		}
	}
}

func replayFloatPtr(value float64) *float64 { return &value }

func TestRouterReplayQueriesOlderSessionBeforeLoadingBodies(t *testing.T) {
	calls := 0
	storage := &replayQueryTestStore{page: func(filters store.QueryFilters, limit, offset int, details bool) (store.RecordPage, error) {
		calls++
		if filters.SessionID != "old-session" || limit != math.MaxInt || offset != 0 || !details || !filters.RecipeSet || filters.Recipe != "evaluation" {
			t.Fatalf("session query=%+v limit=%d details=%v", filters, limit, details)
		}
		count := min(limit, 105-offset)
		records := make([]store.Record, count)
		for i := range records {
			records[i] = store.Record{ID: fmt.Sprintf("old-%03d", 105-offset-i), SessionID: filters.SessionID}
		}
		return store.RecordPage{Records: records, Offset: offset, Total: 105}, nil
	}}
	router := &OpenAIRouter{ReplayRecorder: routerreplay.NewRecorder(storage), ReplayStoreShared: true}
	records, err := router.queryRouterReplaySession("old-session", replayStringPtr("evaluation"))
	if err != nil || len(records) != 105 || calls != 1 || records[104].ID != "old-001" {
		t.Fatalf("older trajectory count=%d calls=%d err=%v", len(records), calls, err)
	}
}

func TestRouterReplayIsolatedDatabasePageHasNextOffset(t *testing.T) {
	readers := make(map[string]*routerreplay.Recorder)
	for _, name := range []string{"a", "b"} {
		storage := &replayQueryTestStore{page: func(_ store.QueryFilters, limit, offset int, details bool) (store.RecordPage, error) {
			if limit != 5 || offset != 0 || details {
				t.Fatalf("isolated prefix must stay bounded and body-free: %d %d %v", limit, offset, details)
			}
			records := make([]store.Record, 5)
			for i := range records {
				records[i].ID = fmt.Sprintf("%s-%d", name, i)
			}
			return store.RecordPage{Records: records, Total: 12230}, nil
		}}
		readers[name] = routerreplay.NewRecorder(storage)
	}
	router := &OpenAIRouter{ReplayRecorders: readers}
	page, err := router.queryRouterReplayPage(routerReplayListQuery{limit: 5})
	if err != nil || page.Total != 24460 || !page.HasMore || page.NextOffset == nil || *page.NextOffset != 5 {
		t.Fatalf("isolated page=%+v err=%v", page, err)
	}
}

func TestRouterReplayIsolatedDetailRetainsSourceForDuplicateIDs(t *testing.T) {
	readers := make(map[string]*routerreplay.Recorder)
	for _, name := range []string{"alpha", "beta"} {
		record := store.Record{ID: "same-id", Recipe: name, RequestBody: name + "-private-body"}
		memory := store.NewMemoryStore(2, 0)
		if _, err := memory.Add(context.Background(), record); err != nil {
			t.Fatal(err)
		}
		storage := &replayQueryTestStore{MemoryStore: memory, page: func(store.QueryFilters, int, int, bool) (store.RecordPage, error) {
			return store.RecordPage{Records: []store.Record{record}, Total: 1}, nil
		}}
		readers[name] = routerreplay.NewRecorder(storage)
	}
	router := &OpenAIRouter{ReplayRecorders: readers}
	page, err := router.queryRouterReplayPage(routerReplayListQuery{limit: 5, showDetails: true})
	if err != nil || len(page.Data) != 2 {
		t.Fatalf("page=%+v err=%v", page, err)
	}
	for i, name := range []string{"alpha", "beta"} {
		if page.Data[i].Recipe != name || page.Data[i].RequestBody != name+"-private-body" {
			t.Fatalf("record crossed store boundary: %+v", page.Data[i])
		}
	}
}
