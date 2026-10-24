package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func assertUnexecuted(t *testing.T, decoder *json.Decoder, id, reason string) {
	t.Helper()
	var got map[string]any
	if err := decoder.Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got["schema"] != "jev-research-not-executed.v1" || got["execution_status"] != "not_executed" || got["id"] != id || got["attempts"] != float64(0) || got["reason"] != reason {
		t.Fatalf("unexpected unexecuted record: %+v", got)
	}
	for _, key := range []string{"request", "raw_response", "elapsed_ms", "http_status", "contract_valid", "correct"} {
		if _, exists := got[key]; exists {
			t.Fatalf("unexecuted case has fabricated measurement %s", key)
		}
	}
	t.Logf("case=%s status=%s attempts=%.0f reason=%s", got["id"], got["execution_status"], got["attempts"], got["reason"])
}

func TestCollectRecordsUnexecutedTail(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := validResponse
		if calls.Add(1) == 2 {
			body = strings.Replace(body, `"coding":0.8`, `"coding":0.3`, 1)
		}
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()
	a, err := newAdapter(server.URL, "test-key", fixtureQuestion(), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer a.client.Close()
	var cases []testCase
	for _, id := range []string{"one", "two", "three", "four", "five", "six"} {
		cases = append(cases, testCase{ID: id, Group: "pilot", State: "debug this"})
	}
	var output bytes.Buffer
	err = collect(context.Background(), a, cases, runOptions{Revision: "draft", Timeout: time.Second}, "data", "question", json.NewEncoder(&output))
	if err == nil || calls.Load() != 2 {
		t.Fatalf("expected failure after two calls, calls=%d err=%v", calls.Load(), err)
	}
	t.Logf("MOCK ONLY: planned=%d local_HTTP_calls=%d; %v", len(cases), calls.Load(), err)
	decoder := json.NewDecoder(&output)
	for i := 0; i < 2; i++ {
		var got record
		if err := decoder.Decode(&got); err != nil {
			t.Fatal(err)
		}
		if got.ID != cases[i].ID || got.Attempts != 1 || got.Valid != (i == 0) {
			t.Fatalf("unexpected executed record: %+v", got)
		}
		t.Logf("case=%s attempts=%d contract_valid=%t error=%q", got.ID, got.Attempts, got.Valid, got.Error)
	}
	for _, c := range cases[2:] {
		assertUnexecuted(t, decoder, c.ID, "stopped_after_failure:two")
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		t.Fatalf("expected exactly six records, got %v", err)
	}
}

func TestCollectCanceledBeforeFirstCall(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var output bytes.Buffer
	// A nil adapter proves cancellation records do not require any API client.
	err := collect(ctx, nil, []testCase{{ID: "one"}, {ID: "two"}}, runOptions{}, "data", "question", json.NewEncoder(&output))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation, got %v", err)
	}
	decoder := json.NewDecoder(&output)
	assertUnexecuted(t, decoder, "one", "context_stopped")
	assertUnexecuted(t, decoder, "two", "context_stopped")
}
