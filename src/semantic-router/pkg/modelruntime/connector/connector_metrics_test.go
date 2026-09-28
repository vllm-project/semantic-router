package connector

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func metricsTestOptions() Options {
	return Options{
		AttemptTimeout:   time.Second,
		MaxRetries:       1,
		MaxRequestBytes:  1 << 20,
		MaxResponseBytes: 1 << 20,
		MaxErrorBytes:    1 << 10,
	}
}

// Every call is counted once under its outcome and each retried attempt as a
// retry, so a remote that flaps under the retry budget is distinguishable
// from one that is healthy or one that is down.
func TestDoRecordsOutcomeAndRetries(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()

	client, err := New(server.URL, nil, metricsTestOptions())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	operation := Operation{Name: "metrics_test_retry", Method: http.MethodPost, Path: "/classify", RetrySafe: true}

	successBefore := testutil.ToFloat64(metrics.RemoteConnectorRequestsTotal.WithLabelValues(operation.Name, metrics.RemoteConnectorOutcomeSuccess))
	retriesBefore := testutil.ToFloat64(metrics.RemoteConnectorRetriesTotal.WithLabelValues(operation.Name))

	if _, err := client.Do(context.Background(), operation, []byte(`{}`)); err != nil {
		t.Fatalf("Do after one retry: %v", err)
	}
	if calls != 2 {
		t.Fatalf("server saw %d calls, want 2 (one failure, one retry)", calls)
	}
	if got := testutil.ToFloat64(metrics.RemoteConnectorRequestsTotal.WithLabelValues(operation.Name, metrics.RemoteConnectorOutcomeSuccess)); got != successBefore+1 {
		t.Fatalf("success outcome = %v, want %v", got, successBefore+1)
	}
	if got := testutil.ToFloat64(metrics.RemoteConnectorRetriesTotal.WithLabelValues(operation.Name)); got != retriesBefore+1 {
		t.Fatalf("retries = %v, want %v", got, retriesBefore+1)
	}
}

// A failure is counted under the connector's own error kind, not a generic
// "error", so a dashboard can tell a bad request from a dead remote.
func TestDoRecordsTheErrorKindAsOutcome(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer server.Close()

	client, err := New(server.URL, nil, metricsTestOptions())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	operation := Operation{Name: "metrics_test_status", Method: http.MethodPost, Path: "/classify"}

	before := testutil.ToFloat64(metrics.RemoteConnectorRequestsTotal.WithLabelValues(operation.Name, string(KindStatus)))
	if _, err := client.Do(context.Background(), operation, []byte(`{}`)); err == nil {
		t.Fatal("expected a status error from a 400")
	}
	if got := testutil.ToFloat64(metrics.RemoteConnectorRequestsTotal.WithLabelValues(operation.Name, string(KindStatus))); got != before+1 {
		t.Fatalf("status outcome = %v, want %v", got, before+1)
	}
}
