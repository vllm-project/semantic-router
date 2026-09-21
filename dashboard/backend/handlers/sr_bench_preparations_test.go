package handlers

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSRBenchPreparationProxyPreservesAsynchronousJob(t *testing.T) {
	const body = `{"benchmark":"mmlu-pro","profile":"smoke","seed":42}`
	const job = `{"preparation":{"id":"prep-0123456789abcdef0123456789abcdef","status":"queued"}}`
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		actual, _ := io.ReadAll(r.Body)
		if r.Method != http.MethodPost || r.URL.Path != SRBenchAPIPath+"/dataset-preparations" || string(actual) != body {
			t.Error("preparation request changed")
		}
		w.WriteHeader(http.StatusAccepted)
		_, _ = io.WriteString(w, job)
	}))
	defer upstream.Close()
	for _, readonly := range []bool{false, true} {
		handler, err := NewSRBenchHandler(upstream.URL, "service-secret", readonly)
		if err != nil {
			t.Fatal(err)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, srBenchTestRequest(http.MethodPost, SRBenchAPIPath+"/dataset-preparations", body))
		if readonly {
			if response.Code != http.StatusForbidden {
				t.Fatalf("readonly preparation accepted: %d", response.Code)
			}
		} else if response.Code != http.StatusAccepted || response.Body.String() != job {
			t.Fatalf("job changed: status=%d body=%s", response.Code, response.Body.String())
		}
	}
	if calls != 1 {
		t.Fatalf("preparation forwarded %d times", calls)
	}
}

func TestSRBenchPreparationsRequireExactRoutesAndMethods(t *testing.T) {
	const id = "prep-0123456789abcdef0123456789abcdef"
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `{}`)
	}))
	defer upstream.Close()
	handler, err := NewSRBenchHandler(upstream.URL, "service-secret", true)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct {
		method, path string
		status       int
	}{
		{http.MethodGet, "/dataset-preparations", http.StatusOK},
		{http.MethodGet, "/dataset-preparations/options", http.StatusOK},
		{http.MethodGet, "/dataset-preparations/" + id, http.StatusOK},
		{http.MethodPost, "/dataset-preparations/options", http.StatusMethodNotAllowed},
		{http.MethodPost, "/dataset-preparations/" + id, http.StatusMethodNotAllowed},
		{http.MethodDelete, "/dataset-preparations/" + id, http.StatusMethodNotAllowed},
		{http.MethodGet, "/dataset-preparations/" + strings.ToUpper(id), http.StatusNotFound},
		{http.MethodGet, "/dataset-preparations/" + id[:len(id)-1], http.StatusNotFound},
		{http.MethodGet, "/dataset-preparations/" + id + "/retry", http.StatusNotFound},
		{http.MethodGet, "/dataset-preparations/prep-%2fetc", http.StatusNotFound},
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, srBenchTestRequest(item.method, SRBenchAPIPath+item.path, ""))
		if response.Code != item.status {
			t.Fatalf("%s %s returned %d, want %d", item.method, item.path, response.Code, item.status)
		}
	}
	if calls != 3 {
		t.Fatalf("unexpected preparation calls: %d", calls)
	}
}
