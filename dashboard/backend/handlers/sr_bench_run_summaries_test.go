package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

type summaryUpstream struct {
	mu          sync.Mutex
	runs        []map[string]any
	reportReads map[string]int
	failReports bool
}

func (u *summaryUpstream) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if r.Header.Get("Authorization") != "Bearer service-secret" || r.Header.Get("X-SR-Bench-Actor-ID") != "owner-123" {
		http.Error(w, `{"error":"unauthenticated"}`, http.StatusUnauthorized)
		return
	}
	if r.URL.Path == SRBenchAPIPath+"/runs" {
		_ = json.NewEncoder(w).Encode(map[string]any{"runs": u.runs, "next": "kept"})
		return
	}
	id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, SRBenchAPIPath+"/runs/"), "/report")
	u.reportReads[id]++
	if u.failReports {
		http.Error(w, `{"error":"report unavailable"}`, http.StatusServiceUnavailable)
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"summary": map[string]any{
			"wall_time_s": 12,
			"targets": []any{map[string]any{
				"id": "single", "total": 2, "completed": 2, "scored": 2, "failed": 0, "pending": 0,
				"accuracy": 0.5, "macro_accuracy": 0.5, "cost_usd": 0.25, "total_spend_usd": 0.3,
				"tokens": map[string]int{"input_tokens": 9}, "selected_models": map[string]int{"m": 2},
			}},
		},
		"provenance": map[string]any{"private": "kept upstream"},
	})
}

func listedTestRun(id, status, mode, updatedAt string) map[string]any {
	return map[string]any{
		"id": id, "status": status, "updated_at": updatedAt,
		"manifest": map[string]any{"name": id, "mode": mode},
		"progress": map[string]int{"total": 2, "completed": 2, "failed": 0},
	}
}

func readRunList(t *testing.T, handler *SRBenchHandler) map[string]map[string]any {
	t.Helper()
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, srBenchTestRequest(http.MethodGet, SRBenchAPIPath+"/runs", ""))
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var document struct {
		Runs []map[string]any `json:"runs"`
		Next string           `json:"next"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &document); err != nil {
		t.Fatal(err)
	}
	if document.Next != "kept" {
		t.Fatalf("top-level fields were not preserved: %s", response.Body.String())
	}
	byID := map[string]map[string]any{}
	for _, run := range document.Runs {
		byID[run["id"].(string)] = run
	}
	return byID
}

func newSummaryHandler(t *testing.T, upstream *summaryUpstream) *SRBenchHandler {
	t.Helper()
	server := httptest.NewServer(upstream)
	t.Cleanup(server.Close)
	handler, err := NewSRBenchHandler(server.URL, "service-secret", false)
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func TestSRBenchRunListAttachesCompactSummariesToFinishedLiveRuns(t *testing.T) {
	upstream := &summaryUpstream{reportReads: map[string]int{}, runs: []map[string]any{
		listedTestRun("done", "completed", "live", "t1"),
		listedTestRun("broken", "failed", "live", "t1"),
		listedTestRun("active", "running", "live", "t1"),
		listedTestRun("preview", "completed", "preview", "t1"),
		func() map[string]any {
			run := listedTestRun("own", "completed", "live", "t1")
			run["summary"] = map[string]any{"targets": []any{}}
			return run
		}(),
	}}
	runs := readRunList(t, newSummaryHandler(t, upstream))
	for _, id := range []string{"done", "broken"} {
		summary, ok := runs[id]["summary"].(map[string]any)
		if !ok {
			t.Fatalf("%s has no summary: %v", id, runs[id])
		}
		target := summary["targets"].([]any)[0].(map[string]any)
		if target["macro_accuracy"] != 0.5 || target["total_spend_usd"] != 0.3 || target["pending"] != float64(0) {
			t.Fatalf("%s summary lost scored fields: %v", id, target)
		}
		if _, leaked := target["tokens"]; leaked {
			t.Fatalf("%s summary carried report usage: %v", id, target)
		}
		if summary["wall_time_s"] != float64(12) {
			t.Fatalf("%s summary lost wall time: %v", id, summary)
		}
	}
	for _, id := range []string{"active", "preview"} {
		if _, ok := runs[id]["summary"]; ok {
			t.Fatalf("%s must not be enriched: %v", id, runs[id])
		}
	}
	if len(runs["own"]["summary"].(map[string]any)["targets"].([]any)) != 0 {
		t.Fatal("a service-provided summary was replaced")
	}
	if upstream.reportReads["active"]+upstream.reportReads["preview"]+upstream.reportReads["own"] != 0 {
		t.Fatalf("unexpected report reads: %v", upstream.reportReads)
	}
}

func TestSRBenchRunListCachesSummariesUntilTheRunChanges(t *testing.T) {
	upstream := &summaryUpstream{reportReads: map[string]int{}, runs: []map[string]any{
		listedTestRun("done", "completed", "live", "t1"),
	}}
	handler := newSummaryHandler(t, upstream)
	readRunList(t, handler)
	runs := readRunList(t, handler)
	if upstream.reportReads["done"] != 1 || runs["done"]["summary"] == nil {
		t.Fatalf("cached summary not reused: reads=%v run=%v", upstream.reportReads, runs["done"])
	}
	upstream.mu.Lock()
	upstream.runs[0]["updated_at"] = "t2"
	upstream.mu.Unlock()
	readRunList(t, handler)
	if upstream.reportReads["done"] != 2 {
		t.Fatalf("a changed run reused its old summary: reads=%v", upstream.reportReads)
	}
}

func TestSRBenchRunListKeepsRunsWhenReportsFailAndBoundsReads(t *testing.T) {
	runs := make([]map[string]any, 0, maxSummaryReadsPerList+5)
	for index := 0; index < maxSummaryReadsPerList+5; index++ {
		runs = append(runs, listedTestRun(fmt.Sprintf("run-%d", index), "completed", "live", "t1"))
	}
	upstream := &summaryUpstream{reportReads: map[string]int{}, runs: runs, failReports: true}
	listed := readRunList(t, newSummaryHandler(t, upstream))
	if len(listed) != len(runs) {
		t.Fatalf("runs were dropped: %d of %d", len(listed), len(runs))
	}
	for id, run := range listed {
		if _, ok := run["summary"]; ok {
			t.Fatalf("%s got a summary from a failed report", id)
		}
	}
	total := 0
	for _, reads := range upstream.reportReads {
		total += reads
	}
	if total != maxSummaryReadsPerList {
		t.Fatalf("report reads=%d, want %d", total, maxSummaryReadsPerList)
	}
}
