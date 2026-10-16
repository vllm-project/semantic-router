package handlers

import (
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestSRBenchDatasetCompositionIsExplicitAndReadonlySafe(t *testing.T) {
	const path = SRBenchAPIPath + "/datasets/compose"
	body := `{"dataset_ids":["` + strings.Repeat("a", 64) + `"],"benchmarks":["mmlu-pro"]}`
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		actual, _ := io.ReadAll(r.Body)
		if r.Method != http.MethodPost || r.URL.Path != path || string(actual) != body {
			t.Errorf("composition request changed: %s %s %s", r.Method, r.URL.Path, actual)
		}
		_, _ = w.Write([]byte(`{"status":"prepared"}`))
	}))
	defer upstream.Close()
	for _, readonly := range []bool{false, true} {
		handler, err := NewSRBenchHandler(upstream.URL, "service-secret", readonly)
		if err != nil {
			t.Fatal(err)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, srBenchTestRequest(http.MethodPost, path, body))
		want := http.StatusOK
		if readonly {
			want = http.StatusForbidden
		}
		if response.Code != want {
			t.Fatalf("readonly=%v status=%d want=%d", readonly, response.Code, want)
		}
		response = httptest.NewRecorder()
		handler.ServeHTTP(response, srBenchTestRequest(http.MethodGet, path, ""))
		if response.Code != http.StatusMethodNotAllowed {
			t.Fatalf("composition accepted a read: %d", response.Code)
		}
	}
	if calls != 1 {
		t.Fatalf("composition forwarded %d times", calls)
	}
}

func TestSRBenchDatasetReadsPreserveFiltersAndRequireFrozenIDs(t *testing.T) {
	id := strings.Repeat("a1", 32)
	paths := []string{
		SRBenchAPIPath + "/datasets/" + id,
		SRBenchAPIPath + "/datasets/" + id + "/cases?cursor=20&limit=10&benchmark=mmlu-pro&category=computer%20science&q=logic",
	}
	var forwarded []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		forwarded = append(forwarded, r.URL.RequestURI())
		if r.Method != http.MethodGet || r.Header.Get("X-SR-Bench-Actor-ID") != "owner-123" {
			t.Error("dataset read lost its method or authenticated actor")
		}
		_, _ = w.Write([]byte(`{"cases":[],"next_cursor":null}`))
	}))
	defer upstream.Close()
	handler, err := NewSRBenchHandler(upstream.URL, "service-secret", true)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range paths {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, srBenchTestRequest(http.MethodGet, path, ""))
		if response.Code != http.StatusOK {
			t.Fatalf("dataset read failed: %d %s", response.Code, response.Body.String())
		}
	}
	if !reflect.DeepEqual(forwarded, paths) {
		t.Fatalf("filters changed: %v", forwarded)
	}
	for _, item := range []struct {
		path, method string
		status       int
	}{
		{paths[0], http.MethodPost, http.StatusMethodNotAllowed},
		{paths[1], http.MethodDelete, http.StatusMethodNotAllowed},
		{SRBenchAPIPath + "/datasets/" + strings.ToUpper(id), http.MethodGet, http.StatusNotFound},
		{SRBenchAPIPath + "/datasets/" + id[:63], http.MethodGet, http.StatusNotFound},
		{SRBenchAPIPath + "/datasets/" + id + "0", http.MethodGet, http.StatusNotFound},
		{SRBenchAPIPath + "/datasets/" + id + "/cases/extra", http.MethodGet, http.StatusNotFound},
		{SRBenchAPIPath + "/datasets/" + id + "/export", http.MethodGet, http.StatusNotFound},
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, srBenchTestRequest(item.method, item.path, ""))
		if response.Code != item.status {
			t.Fatalf("%s %s returned %d, want %d", item.method, item.path, response.Code, item.status)
		}
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, paths[0], nil))
	if response.Code != http.StatusUnauthorized || len(forwarded) != len(paths) {
		t.Fatalf("unexpected dataset forwarding: status=%d requests=%v", response.Code, forwarded)
	}
}
