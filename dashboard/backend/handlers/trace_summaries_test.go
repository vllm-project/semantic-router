package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	traceSummaryTestID    = "4bf92f3577b34da6a3ce929d0e0e4736"
	traceSummaryNewerID   = "5c3d0a0f7e2b4c19a8d6e1f203948576"
	traceSummaryTestQuery = "service=vllm-sr"
)

var traceSummaryTestNow = time.Date(2026, 11, 18, 7, 0, 0, 0, time.UTC)

// The request trace covers each summary rule: a FOLLOWS_FROM span that starts
// before the root, an orphan and a self-reference that start after it, a span
// that ends after the root, a negative duration, two error marks, a tag
// payload, and a malformed span. A partial copy of it comes before and after
// it, beside a newer one-span trace and three unreadable entries.
const jaegerSearchFixture = `{
  "data": [
    {"traceID": "4bf92f3577b34da6a3ce929d0e0e4736", "spans": [
      {"spanID": "root", "operationName": "semantic_router.request", "references": [], "startTime": 1000000, "duration": 900000, "processID": "p1", "tags": []}
    ], "processes": {"p1": {"serviceName": "vllm-sr"}}},
    {"traceID": "4bf92f3577b34da6a3ce929d0e0e4736", "spans": [
      {"traceID": "4bf92f3577b34da6a3ce929d0e0e4736", "spanID": "root", "operationName": "semantic_router.request", "references": [],
       "startTime": 1000000, "duration": 900000, "processID": "p1",
       "tags": [{"key": "span.kind", "type": "string", "value": "server"}, {"key": "error", "type": "bool", "value": false}],
       "logs": [{"timestamp": 1100000, "fields": [{"key": "event", "type": "string", "value": "routing.decision"}]}]},
      {"spanID": "decide", "operationName": "semantic_router.decision.evaluation", "references": [{"refType": "CHILD_OF", "spanID": "root"}],
       "startTime": 1100000, "duration": 100000, "processID": "p1", "tags": []},
      {"spanID": "upstream", "operationName": "semantic_router.upstream.request", "references": [{"refType": "CHILD_OF", "spanID": "root"}],
       "startTime": 1300000, "duration": 800000, "processID": "p2",
       "tags": [{"key": "error", "type": "bool", "value": true},
                {"key": "http.request.body", "type": "string", "value": "{\"messages\":[{\"role\":\"user\",\"content\":\"Write a haiku about tracing.\"}]}"}]},
      {"spanID": "model", "operationName": "POST /v1/chat/completions", "references": [{"refType": "CHILD_OF", "spanID": "upstream"}],
       "startTime": 1350000, "duration": 400000, "processID": "p2", "tags": [{"key": "otel.status_code", "type": "string", "value": "error"}]},
      {"spanID": "prefetch", "operationName": "envoy.prefetch", "references": [{"refType": "FOLLOWS_FROM", "spanID": "upstream"}],
       "startTime": 950000, "duration": 20000, "processID": "p3", "tags": [{"key": "error", "type": "string", "value": "false"}]},
      {"spanID": "orphan", "operationName": "semantic_router.cache.write", "references": [{"refType": "CHILD_OF", "spanID": "gone"}],
       "startTime": 1200000, "duration": 10000, "processID": "p2", "tags": []},
      {"spanID": "self", "operationName": "semantic_router.loop", "references": [{"refType": "CHILD_OF", "spanID": "self"}],
       "startTime": 1250000, "duration": 10000, "processID": "p2"},
      {"spanID": "skewed", "operationName": "semantic_router.clock", "references": [{"refType": "CHILD_OF", "spanID": "root"}],
       "startTime": 1500000, "duration": -5, "processID": "p1"},
      {"spanID": "malformed", "operationName": "no timing", "processID": "p1"}
    ], "processes": {
      "p1": {"serviceName": "vllm-sr", "tags": [{"key": "hostname", "type": "string", "value": "router-0"}]},
      "p2": {"serviceName": "qwen3-backend", "tags": []},
      "p3": {"serviceName": "envoy", "tags": []}
    }, "warnings": null},
    {"traceID": "5c3d0a0f7e2b4c19a8d6e1f203948576", "spans": [
      {"spanID": "lone", "operationName": "semantic_router.cache.lookup", "references": [{"refType": "CHILD_OF", "spanID": "gone"}],
       "startTime": 5000000, "duration": 250000, "processID": "p1", "tags": []}
    ], "processes": {"p1": {"serviceName": "vllm-sr"}}},
    {"traceID": "4bf92f3577b34da6a3ce929d0e0e4736", "spans": [
      {"spanID": "root", "operationName": "semantic_router.request", "references": [], "startTime": 1000000, "duration": 900000, "processID": "p1"}
    ], "processes": {"p1": {"serviceName": "vllm-sr"}}},
    {"traceID": "empty", "spans": [{"spanID": "x"}]},
    {"traceID": 42, "spans": "not a list"},
    null
  ],
  "total": 0, "limit": 0, "offset": 0, "errors": null
}`

var wantJaegerSearchSummaries = []TraceSummary{
	{
		TraceID: traceSummaryNewerID, RootOperation: "semantic_router.cache.lookup", RootService: "vllm-sr",
		Services:  []TraceServiceSummary{{Name: "vllm-sr", SpanCount: 1}},
		SpanCount: 1, StartTimeMicros: 5_000_000, DurationMicros: 250_000,
	},
	{
		TraceID: traceSummaryTestID, RootOperation: "semantic_router.request", RootService: "vllm-sr",
		Services: []TraceServiceSummary{
			{Name: "vllm-sr", SpanCount: 3},
			{Name: "qwen3-backend", SpanCount: 4},
			{Name: "envoy", SpanCount: 1},
		},
		SpanCount: 8, ErrorCount: 2, StartTimeMicros: 950_000, DurationMicros: 1_150_000,
	},
}

type recordedJaegerSearch struct {
	mu       sync.Mutex
	path     string
	query    url.Values
	requests int
}

func (recorded *recordedJaegerSearch) record(r *http.Request) {
	recorded.mu.Lock()
	defer recorded.mu.Unlock()
	recorded.path, recorded.query = r.URL.Path, r.URL.Query()
	recorded.requests++
}

func (recorded *recordedJaegerSearch) count() int {
	recorded.mu.Lock()
	defer recorded.mu.Unlock()
	return recorded.requests
}

func newFakeJaeger(t *testing.T, contentType, body string) (*httptest.Server, *recordedJaegerSearch) {
	t.Helper()
	recorded := &recordedJaegerSearch{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		recorded.record(r)
		w.Header().Set("Content-Type", contentType)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)
	return server, recorded
}

func serveTraceSummaries(t *testing.T, options traceSummariesOptions, query string) *httptest.ResponseRecorder {
	t.Helper()
	if options.now == nil {
		options.now = func() time.Time { return traceSummaryTestNow }
	}
	recorder := httptest.NewRecorder()
	newTraceSummariesHandler(options)(recorder, httptest.NewRequest(http.MethodGet, "/api/observability/traces/summaries?"+query, nil))
	return recorder
}

func decodeTraceSummaries(t *testing.T, recorder *httptest.ResponseRecorder) TraceSummariesResponse {
	t.Helper()
	if recorder.Code != http.StatusOK || recorder.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("status = %d, content type = %q, body = %s", recorder.Code, recorder.Header().Get("Content-Type"), recorder.Body)
	}
	var response TraceSummariesResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode %s: %v", recorder.Body, err)
	}
	return response
}

func expectTraceSummariesError(t *testing.T, recorder *httptest.ResponseRecorder, status int, code string) TraceSummariesErrorResponse {
	t.Helper()
	var response TraceSummariesErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode %s: %v", recorder.Body, err)
	}
	if recorder.Code != status || response.Error != code || response.Message == "" ||
		recorder.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("status = %d, body = %+v, want %d %s with a message", recorder.Code, response, status, code)
	}
	return response
}

func TestTraceSummariesSummarizeEachTraceOnceNewestFirst(t *testing.T) {
	jaeger, _ := newFakeJaeger(t, "application/json", jaegerSearchFixture)
	response := decodeTraceSummaries(t, serveTraceSummaries(t, traceSummariesOptions{jaegerURL: jaeger.URL}, traceSummaryTestQuery))
	if !reflect.DeepEqual(response.Data, wantJaegerSearchSummaries) {
		t.Fatalf("summaries =\n%+v\nwant\n%+v", response.Data, wantJaegerSearchSummaries)
	}
	if response.Limit != defaultTraceSummaryLimit || response.Total != 2 {
		t.Fatalf("limit = %d, total = %d", response.Limit, response.Total)
	}

	limited := decodeTraceSummaries(t, serveTraceSummaries(t, traceSummariesOptions{jaegerURL: jaeger.URL}, traceSummaryTestQuery+"&limit=1"))
	if len(limited.Data) != 1 || limited.Data[0].TraceID != traceSummaryNewerID || limited.Limit != 1 || limited.Total != 1 {
		t.Fatalf("limited summaries = %+v", limited)
	}
}

func TestTraceSummariesRootFallsBackToEarliestSpanInAReferenceCycle(t *testing.T) {
	summary, ok := summarizeJaegerTrace(jaegerTrace{
		TraceID: traceSummaryTestID,
		Spans: []jaegerSpan{
			{SpanID: "b", OperationName: "second", StartTime: ptrInt64(20), Duration: ptrInt64(5), References: []jaegerReference{{RefType: "CHILD_OF", SpanID: "a"}}},
			{SpanID: "a", OperationName: "first", StartTime: ptrInt64(10), Duration: ptrInt64(5), References: []jaegerReference{{RefType: "CHILD_OF", SpanID: "b"}}},
		},
	})
	if !ok || summary.RootOperation != "first" || summary.RootService != "" || summary.DurationMicros != 15 ||
		!reflect.DeepEqual(summary.Services, []TraceServiceSummary{{Name: "", SpanCount: 2}}) {
		t.Fatalf("summary = %+v, ok = %v", summary, ok)
	}
	if _, ok := summarizeJaegerTrace(jaegerTrace{Spans: []jaegerSpan{{SpanID: "a", StartTime: ptrInt64(1), Duration: ptrInt64(1)}}}); ok {
		t.Fatal("a trace without an ID was summarized")
	}
}

func ptrInt64(value int64) *int64 { return &value }

func TestTraceSummariesForwardTheSearchAsJaegerReadsIt(t *testing.T) {
	jaeger, recorded := newFakeJaeger(t, "application/json; charset=utf-8", `{"data":null,"errors":null}`)
	options := traceSummariesOptions{jaegerURL: jaeger.URL + "/jaeger/"}
	query := url.Values{
		"service": {"vllm-sr"}, "operation": {"semantic_router.request"}, "lookback": {"6h"},
		"limit": {"50"}, "minDuration": {"250ms"}, "maxDuration": {"1.5s"},
	}
	response := decodeTraceSummaries(t, serveTraceSummaries(t, options, query.Encode()))
	if response.Data == nil || len(response.Data) != 0 || response.Limit != 50 || response.Total != 0 {
		t.Fatalf("response = %+v, want an empty list", response)
	}
	end := traceSummaryTestNow.UnixMicro()
	want := url.Values{
		"service": {"vllm-sr"}, "operation": {"semantic_router.request"},
		"start": {strconv.FormatInt(end-(6*time.Hour).Microseconds(), 10)}, "end": {strconv.FormatInt(end, 10)},
		"limit": {"50"}, "minDuration": {"250ms"}, "maxDuration": {"1.5s"},
	}
	if recorded.path != "/jaeger/api/traces" || !reflect.DeepEqual(recorded.query, want) {
		t.Fatalf("Jaeger got %s?%v, want /jaeger/api/traces?%v", recorded.path, recorded.query, want)
	}

	decodeTraceSummaries(t, serveTraceSummaries(t, traceSummariesOptions{jaegerURL: jaeger.URL}, traceSummaryTestQuery))
	want = url.Values{
		"service": {"vllm-sr"},
		"start":   {strconv.FormatInt(end-time.Hour.Microseconds(), 10)}, "end": {strconv.FormatInt(end, 10)},
		"limit": {"20"},
	}
	if recorded.path != "/api/traces" || !reflect.DeepEqual(recorded.query, want) {
		t.Fatalf("default search sent %s?%v, want /api/traces?%v", recorded.path, recorded.query, want)
	}
}

func TestTraceSummariesRejectInvalidSearchesBeforeAskingJaeger(t *testing.T) {
	jaeger, recorded := newFakeJaeger(t, "application/json", `{"data":[]}`)
	for _, query := range []string{
		"",
		"service=%20%20",
		"service=vllm%0Asr",
		"service=vllm-sr&operation=" + strings.Repeat("o", maxTraceSearchNameBytes+1),
		"service=vllm-sr&limit=0",
		"service=vllm-sr&limit=101",
		"service=vllm-sr&limit=ten",
		"service=vllm-sr&lookback=7d",
		"service=vllm-sr&lookback=-1h",
		"service=vllm-sr&lookback=0s",
		"service=vllm-sr&lookback=721h",
		"service=vllm-sr&minDuration=fast",
		"service=vllm-sr&minDuration=-1s",
		"service=vllm-sr&maxDuration=100",
		"service=vllm-sr&minDuration=1s&maxDuration=250ms",
	} {
		t.Run(query, func(t *testing.T) {
			expectTraceSummariesError(t, serveTraceSummaries(t, traceSummariesOptions{jaegerURL: jaeger.URL}, query), http.StatusBadRequest, "invalid_request")
		})
	}
	if recorded.count() != 0 {
		t.Fatalf("Jaeger got %d searches for invalid input", recorded.count())
	}
}

func TestTraceSummariesExplainEachJaegerFailure(t *testing.T) {
	closed := httptest.NewServer(http.NotFoundHandler())
	closedURL := closed.URL
	closed.Close()

	answer := func(status int, contentType, body string) string {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", contentType)
			w.WriteHeader(status)
			_, _ = io.WriteString(w, body)
		}))
		t.Cleanup(server.Close)
		return server.URL
	}
	for _, test := range []struct {
		name, jaegerURL string
		maxBytes        int64
		status          int
		code, message   string
	}{
		{name: "unconfigured", status: http.StatusServiceUnavailable, code: "jaeger_not_configured", message: "TARGET_JAEGER_URL"},
		{name: "not a URL", jaegerURL: "localhost:16686", status: http.StatusServiceUnavailable, code: "jaeger_not_configured", message: "absolute URL"},
		{name: "unreachable", jaegerURL: closedURL, status: http.StatusBadGateway, code: "jaeger_unreachable", message: "Could not reach Jaeger"},
		{name: "page", jaegerURL: answer(http.StatusOK, "text/html", "<!doctype html><title>Jaeger UI</title>"), status: http.StatusBadGateway, code: "jaeger_bad_response", message: "a page instead of trace data"},
		{name: "truncated JSON", jaegerURL: answer(http.StatusOK, "application/json", `{"data":[{"traceID":`), status: http.StatusBadGateway, code: "jaeger_bad_response", message: "could not read"},
		{name: "empty body", jaegerURL: answer(http.StatusOK, "application/json", ""), status: http.StatusBadGateway, code: "jaeger_bad_response", message: "could not read"},
		{name: "not an object", jaegerURL: answer(http.StatusOK, "application/json", `[{"traceID":"a"}]`), status: http.StatusBadGateway, code: "jaeger_bad_response", message: "could not read"},
		{name: "data not a list", jaegerURL: answer(http.StatusOK, "application/json", `{"data":{}}`), status: http.StatusBadGateway, code: "jaeger_bad_response", message: "could not read"},
		{name: "too large", jaegerURL: answer(http.StatusOK, "application/json", jaegerSearchFixture), maxBytes: 256, status: http.StatusBadGateway, code: "jaeger_response_too_large", message: "Show fewer traces"},
		{name: "errors without data", jaegerURL: answer(http.StatusOK, "application/json", `{"data":[],"errors":[{"code":500,"msg":"storage unavailable"}]}`), status: http.StatusBadGateway, code: "jaeger_error", message: "storage unavailable"},
		{name: "rejected", jaegerURL: answer(http.StatusBadRequest, "application/json", `{"data":null,"errors":[{"code":400,"msg":"unable to parse param 'limit'"}]}`), status: http.StatusBadRequest, code: "invalid_request", message: "unable to parse param 'limit'"},
		{name: "failed", jaegerURL: answer(http.StatusInternalServerError, "application/json", `{"data":null,"errors":[{"code":500,"msg":"elasticsearch: no shards"}]}`), status: http.StatusBadGateway, code: "jaeger_error", message: "(HTTP 500): elasticsearch: no shards"},
		{name: "not found", jaegerURL: answer(http.StatusNotFound, "text/plain", "404 page not found"), status: http.StatusBadGateway, code: "jaeger_error", message: "(HTTP 404)"},
		{name: "upstream timeout", jaegerURL: answer(http.StatusGatewayTimeout, "text/plain", "upstream request timeout"), status: http.StatusGatewayTimeout, code: "jaeger_timeout", message: "timed out"},
	} {
		t.Run(test.name, func(t *testing.T) {
			options := traceSummariesOptions{jaegerURL: test.jaegerURL, maxBytes: test.maxBytes}
			response := expectTraceSummariesError(t, serveTraceSummaries(t, options, traceSummaryTestQuery), test.status, test.code)
			if !strings.Contains(response.Message, test.message) {
				t.Fatalf("message = %q, want it to mention %q", response.Message, test.message)
			}
		})
	}
}

func TestTraceSummariesTimeOutWhileJaegerAnswersOrDownloads(t *testing.T) {
	for _, test := range []struct {
		name    string
		partial string
	}{
		{name: "before headers"},
		{name: "during the body", partial: `{"data":[{"traceID":"` + traceSummaryTestID + `","spans":[`},
	} {
		t.Run(test.name, func(t *testing.T) {
			stalled := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if test.partial != "" {
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, test.partial)
					w.(http.Flusher).Flush()
				}
				select {
				case <-r.Context().Done():
				case <-time.After(5 * time.Second):
				}
			}))
			defer stalled.Close()
			options := traceSummariesOptions{jaegerURL: stalled.URL, timeout: 50 * time.Millisecond}
			response := expectTraceSummariesError(t, serveTraceSummaries(t, options, traceSummaryTestQuery), http.StatusGatewayTimeout, "jaeger_timeout")
			if !strings.Contains(response.Message, "within 50ms") {
				t.Fatalf("message = %q", response.Message)
			}
		})
	}
}

func TestTraceSummariesAnswerOnlyReads(t *testing.T) {
	jaeger, recorded := newFakeJaeger(t, "application/json", `{"data":[]}`)
	recorder := httptest.NewRecorder()
	newTraceSummariesHandler(traceSummariesOptions{jaegerURL: jaeger.URL})(recorder, httptest.NewRequest(http.MethodPost, "/api/observability/traces/summaries?"+traceSummaryTestQuery, nil))
	expectTraceSummariesError(t, recorder, http.StatusMethodNotAllowed, "method_not_allowed")
	if recorder.Header().Get("Allow") != "GET, HEAD" || recorded.count() != 0 {
		t.Fatalf("Allow = %q, Jaeger searches = %d", recorder.Header().Get("Allow"), recorded.count())
	}
}
