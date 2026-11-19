package extproc

import (
	"context"
	"fmt"
	"net/url"
	"reflect"
	"testing"
	"time"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// newReplayHarnessRouter holds two Claude Code calls, one Codex call and
// three calls no harness sent, one of them with an agent id alone. The newest
// records are the API client's, as when a scripted client sends most traffic.
func newReplayHarnessRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(20, 0))
	start := time.Unix(1000, 0).UTC()
	for index, record := range []routerreplay.RoutingRecord{
		{ID: "claude-1", Decision: "code", Agent: &store.AgentIdentity{Harness: store.HarnessClaudeCode}},
		{ID: "claude-2", Decision: "chat", Agent: &store.AgentIdentity{Harness: store.HarnessClaudeCode, ID: "subagent"}},
		{ID: "codex-1", Decision: "code", Agent: &store.AgentIdentity{Harness: store.HarnessCodex}},
		{ID: "api-1", Decision: "chat"},
		{ID: "api-2", Decision: "code", Agent: &store.AgentIdentity{ID: "agent-hash"}},
		{ID: "api-3", Decision: "chat"},
	} {
		record.RequestID, record.Timestamp = record.ID, start.Add(time.Duration(index)*time.Second)
		if _, err := recorder.AddRecord(record); err != nil {
			t.Fatal(err)
		}
	}
	return &OpenAIRouter{ReplayRecorder: recorder, ReplayStoreShared: true}
}

func replayHarnessCounts(t *testing.T, body map[string]interface{}) map[string]int {
	t.Helper()
	raw, ok := body["harness_counts"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected harness_counts object, got %#v", body["harness_counts"])
	}
	counts := make(map[string]int, len(raw))
	for harness, count := range raw {
		counts[harness] = int(count.(float64))
	}
	return counts
}

func replayDataIDs(t *testing.T, body map[string]interface{}) []string {
	t.Helper()
	data := mustReplayData(t, body["data"])
	ids := make([]string, len(data))
	for index, record := range data {
		ids[index] = record["id"].(string)
	}
	return ids
}

// The list filters by harness over every record, not a recent page, and
// counts each harness over the records every other filter keeps.
func TestRouterReplayListFiltersAndCountsHarnesses(t *testing.T) {
	router := newReplayHarnessRouter(t)
	everyHarness := map[string]int{store.HarnessClaudeCode: 2, store.HarnessCodex: 1, store.HarnessAPIClient: 3}

	agents := mustReplayListBody(t, router, routerReplayAPIBasePath+"?harness=codex,claude_code&limit=2")
	assertReplayPage(t, agents, 2, 3, 2, 0)
	if ids := replayDataIDs(t, agents); !reflect.DeepEqual(ids, []string{"codex-1", "claude-2"}) {
		t.Fatalf("agent page = %v", ids)
	}
	if counts := replayHarnessCounts(t, agents); !reflect.DeepEqual(counts, everyHarness) {
		t.Fatalf("harness counts under a harness filter = %v, want %v", counts, everyHarness)
	}

	clients := mustReplayListBody(t, router, routerReplayAPIBasePath+"?harness=api_client")
	if ids := replayDataIDs(t, clients); !reflect.DeepEqual(ids, []string{"api-3", "api-2", "api-1"}) {
		t.Fatalf("API client page = %v", ids)
	}

	unfiltered := mustReplayListBody(t, router, routerReplayAPIBasePath+"?limit=1")
	assertIntField(t, unfiltered, "total", 6)
	if counts := replayHarnessCounts(t, unfiltered); !reflect.DeepEqual(counts, everyHarness) {
		t.Fatalf("unfiltered harness counts = %v", counts)
	}

	code := mustReplayListBody(t, router, routerReplayAPIBasePath+"?decision=code&harness=claude_code")
	assertIntField(t, code, "total", 1)
	if counts := replayHarnessCounts(t, code); !reflect.DeepEqual(counts, map[string]int{store.HarnessClaudeCode: 1, store.HarnessCodex: 1, store.HarnessAPIClient: 1}) {
		t.Fatalf("harness counts under a decision filter = %v", counts)
	}

	aggregate := mustReplayListBody(t, router, routerReplayAggregatePath+"?harness=claude_code")
	assertIntField(t, aggregate, "record_count", 2)
}

func TestRouterReplayHarnessFilterIsCanonicalAndChecked(t *testing.T) {
	parse := func(query string) (routerReplayFilters, error) {
		values, err := url.ParseQuery(query)
		if err != nil {
			t.Fatal(err)
		}
		return parseRouterReplayFilters(values)
	}
	repeated, err := parse("harness=codex&harness=api_client,codex")
	if err != nil {
		t.Fatal(err)
	}
	listed, err := parse("harness=" + url.QueryEscape(" api_client , codex "))
	if err != nil || repeated.harness != "api_client,codex" || repeated.cacheKey() != listed.cacheKey() {
		t.Fatalf("harness filters %q and %q (%v) must read alike", repeated.harness, listed.harness, err)
	}
	if repeated.storageFilters().Harness != "api_client,codex" {
		t.Fatalf("storage filters lost the harness: %+v", repeated.storageFilters())
	}

	router := newReplayHarnessRouter(t)
	response := router.handleRouterReplayAPI(context.Background(), "GET", routerReplayAPIBasePath+"?harness=claude")
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != typev3.StatusCode_BadRequest {
		t.Fatalf("unknown harness status = %v", got)
	}
	message := decodeJSONBody(t, response.GetImmediateResponse().Body)["error"].(map[string]interface{})["message"]
	if message != "harness must list one or more of claude_code, codex, opencode, deepseek_harness, cursor, openclaw, api_client" {
		t.Fatalf("unknown harness message = %v", message)
	}
}

// A database store gets the harness filter and answers the counts itself.
func TestRouterReplayDatabasePageCarriesHarnessCounts(t *testing.T) {
	counts := map[string]int{store.HarnessCodex: 977, store.HarnessAPIClient: 51234}
	storage := &replayQueryTestStore{page: func(f store.QueryFilters, l, o int, d bool) (store.RecordPage, error) {
		if f.Harness != "api_client,codex" || l != 25 || o != 0 || d {
			t.Fatalf("query lost the harness filter or page: %+v %d %d %v", f, l, o, d)
		}
		return store.RecordPage{Records: []store.Record{{ID: "newest"}}, Total: 52211, HarnessCounts: counts}, nil
	}}
	router := &OpenAIRouter{ReplayRecorder: routerreplay.NewRecorder(storage), ReplayStoreShared: true}
	body := mustReplayListBody(t, router, routerReplayAPIBasePath+"?harness=codex,api_client&limit=25")
	assertIntField(t, body, "total", 52211)
	if got := replayHarnessCounts(t, body); !reflect.DeepEqual(got, counts) {
		t.Fatalf("database harness counts = %v", got)
	}
}

// Isolated stores without database queries each count their own records.
func TestRouterReplayIsolatedStoresAddHarnessCounts(t *testing.T) {
	readers := make(map[string]*routerreplay.Recorder)
	for index, harness := range []string{store.HarnessClaudeCode, store.HarnessCodex} {
		recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
		for _, agent := range []*store.AgentIdentity{{Harness: harness}, nil} {
			id := fmt.Sprintf("%s-%v", harness, agent != nil)
			if _, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: id, Timestamp: time.Unix(int64(index), 0), Agent: agent}); err != nil {
				t.Fatal(err)
			}
		}
		readers[harness] = recorder
	}
	router := &OpenAIRouter{ReplayRecorders: readers}
	body := mustReplayListBody(t, router, routerReplayAPIBasePath+"?harness=codex")
	assertIntField(t, body, "total", 1)
	want := map[string]int{store.HarnessClaudeCode: 1, store.HarnessCodex: 1, store.HarnessAPIClient: 2}
	if got := replayHarnessCounts(t, body); !reflect.DeepEqual(got, want) {
		t.Fatalf("isolated harness counts = %v, want %v", got, want)
	}
}
