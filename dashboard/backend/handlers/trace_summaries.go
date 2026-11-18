package handlers

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode"
)

const (
	defaultTraceSummaryLimit = 20
	maxTraceSummaryLimit     = 100
	defaultTraceLookback     = time.Hour
	maxTraceLookback         = 720 * time.Hour
	maxTraceSearchNameBytes  = 512
	traceSearchTimeout       = 20 * time.Second
	// The decoder holds one trace at a time, so this bounds the transfer, not
	// memory; live traces run to about 1 MB each, and 100 of them must fit.
	maxTraceSearchBytes     = 256 << 20
	maxJaegerErrorBodyBytes = 64 << 10
	maxJaegerErrorMessage   = 512
)

// TraceSummariesResponse is one summary per trace a Jaeger search found,
// newest first, so a trace list never downloads the traces themselves.
type TraceSummariesResponse struct {
	Data []TraceSummary `json:"data"`
	// Limit is the most traces the search asked Jaeger for.
	Limit int `json:"limit"`
	// Total is the number of summaries in Data.
	Total int `json:"total"`
}

// TraceSummary is what a trace list shows of one trace. TraceID is Jaeger's
// full ID. StartTimeMicros is microseconds since the Unix epoch, and
// DurationMicros runs from the first span's start to the last span's end.
type TraceSummary struct {
	TraceID string `json:"traceID"`
	// RootOperation and RootService describe the earliest span without a
	// parent in the trace, or the earliest span when references form a cycle.
	RootOperation   string                `json:"rootOperation"`
	RootService     string                `json:"rootService"`
	Services        []TraceServiceSummary `json:"services"`
	SpanCount       int                   `json:"spanCount"`
	ErrorCount      int                   `json:"errorCount"`
	StartTimeMicros int64                 `json:"startTimeMicros"`
	DurationMicros  int64                 `json:"durationMicros"`
}

// TraceServiceSummary is one service of a trace and how many of its spans it
// produced. A trace lists the root's service first, then the busiest.
type TraceServiceSummary struct {
	Name      string `json:"name"`
	SpanCount int    `json:"spanCount"`
}

// TraceSummariesErrorResponse explains why no summaries could be returned.
type TraceSummariesErrorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

// TraceSummariesHandler runs a trace search against the Jaeger query service
// at jaegerURL and answers a TraceSummariesResponse. The search takes
// service, operation, lookback, limit, minDuration and maxDuration.
func TraceSummariesHandler(jaegerURL string) http.HandlerFunc {
	return newTraceSummariesHandler(traceSummariesOptions{jaegerURL: jaegerURL})
}

type traceSummariesOptions struct {
	jaegerURL string
	client    *http.Client
	now       func() time.Time
	timeout   time.Duration
	maxBytes  int64
}

func newTraceSummariesHandler(options traceSummariesOptions) http.HandlerFunc {
	if options.client == nil {
		options.client = &http.Client{}
	}
	if options.now == nil {
		options.now = time.Now
	}
	if options.timeout <= 0 {
		options.timeout = traceSearchTimeout
	}
	if options.maxBytes <= 0 {
		options.maxBytes = maxTraceSearchBytes
	}
	endpoint, unconfigured := jaegerSearchEndpoint(options.jaegerURL)
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			writeTraceSummariesError(w, &traceSearchError{http.StatusMethodNotAllowed, "method_not_allowed", "Only GET and HEAD are supported."})
			return
		}
		if unconfigured != nil {
			writeTraceSummariesError(w, unconfigured)
			return
		}
		search, invalid := parseTraceSearch(r.URL.Query())
		if invalid != nil {
			writeTraceSummariesError(w, invalid)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), options.timeout)
		defer cancel()
		summaries, err := searchJaegerTraces(ctx, options, endpoint, search)
		if err != nil {
			if r.Context().Err() != nil {
				return
			}
			failure := classifyTraceSearchFailure(ctx, err, options)
			log.Printf("Trace summaries for service %q failed: %v", search.service, err)
			writeTraceSummariesError(w, failure)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(TraceSummariesResponse{Data: summaries, Limit: search.limit, Total: len(summaries)})
	}
}

// traceSearchError is a failure with the status and code the endpoint answers.
type traceSearchError struct {
	status  int
	code    string
	message string
}

func (e *traceSearchError) Error() string { return e.message }

func writeTraceSummariesError(w http.ResponseWriter, failure *traceSearchError) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(failure.status)
	_ = json.NewEncoder(w).Encode(TraceSummariesErrorResponse{Error: failure.code, Message: failure.message})
}

func jaegerSearchEndpoint(jaegerURL string) (*url.URL, *traceSearchError) {
	if strings.TrimSpace(jaegerURL) == "" {
		return nil, &traceSearchError{
			http.StatusServiceUnavailable, "jaeger_not_configured",
			"Jaeger is not connected to the Dashboard. Set TARGET_JAEGER_URL to the Jaeger query address, for example http://localhost:16686.",
		}
	}
	base, err := url.Parse(jaegerURL)
	if err != nil || base.Host == "" {
		return nil, &traceSearchError{
			http.StatusServiceUnavailable, "jaeger_not_configured",
			"TARGET_JAEGER_URL is not an absolute URL such as http://localhost:16686.",
		}
	}
	return base.JoinPath("api", "traces"), nil
}

type traceSearch struct {
	service     string
	operation   string
	lookback    time.Duration
	limit       int
	minDuration time.Duration
	maxDuration time.Duration
}

func invalidTraceSearch(message string) *traceSearchError {
	return &traceSearchError{http.StatusBadRequest, "invalid_request", message}
}

func parseTraceSearch(query url.Values) (traceSearch, *traceSearchError) {
	search := traceSearch{
		service:   query.Get("service"),
		operation: query.Get("operation"),
		lookback:  defaultTraceLookback,
		limit:     defaultTraceSummaryLimit,
	}
	if strings.TrimSpace(search.service) == "" {
		return search, invalidTraceSearch("service is required.")
	}
	for _, field := range []struct{ name, value string }{{"service", search.service}, {"operation", search.operation}} {
		if len(field.value) > maxTraceSearchNameBytes || strings.ContainsFunc(field.value, unicode.IsControl) {
			return search, invalidTraceSearch(fmt.Sprintf("%s must be at most %d bytes, without control characters.", field.name, maxTraceSearchNameBytes))
		}
	}
	if raw := query.Get("limit"); raw != "" {
		limit, err := strconv.Atoi(raw)
		if err != nil || limit < 1 || limit > maxTraceSummaryLimit {
			return search, invalidTraceSearch(fmt.Sprintf("limit must be a whole number from 1 to %d.", maxTraceSummaryLimit))
		}
		search.limit = limit
	}
	if raw := query.Get("lookback"); raw != "" {
		lookback, err := time.ParseDuration(raw)
		if err != nil || lookback <= 0 || lookback > maxTraceLookback {
			return search, invalidTraceSearch("lookback must be a duration such as 15m, 1h or 168h, at most 720h.")
		}
		search.lookback = lookback
	}
	for _, bound := range []struct {
		name  string
		value *time.Duration
	}{{"minDuration", &search.minDuration}, {"maxDuration", &search.maxDuration}} {
		raw := query.Get(bound.name)
		if raw == "" {
			continue
		}
		duration, err := time.ParseDuration(raw)
		if err != nil || duration < 0 {
			return search, invalidTraceSearch(bound.name + " must be a duration such as 250ms or 1.5s.")
		}
		*bound.value = duration
	}
	if search.maxDuration > 0 && search.maxDuration < search.minDuration {
		return search, invalidTraceSearch("maxDuration must not be shorter than minDuration.")
	}
	return search, nil
}

// jaegerQuery is the search as Jaeger's query API reads it. That API ignores
// lookback and, without start and end, searches its default of the last two days.
func (search traceSearch) jaegerQuery(now time.Time) url.Values {
	query := url.Values{}
	query.Set("service", search.service)
	if search.operation != "" {
		query.Set("operation", search.operation)
	}
	query.Set("start", strconv.FormatInt(now.Add(-search.lookback).UnixMicro(), 10))
	query.Set("end", strconv.FormatInt(now.UnixMicro(), 10))
	query.Set("limit", strconv.Itoa(search.limit))
	if search.minDuration > 0 {
		query.Set("minDuration", search.minDuration.String())
	}
	if search.maxDuration > 0 {
		query.Set("maxDuration", search.maxDuration.String())
	}
	return query
}

var (
	errTraceSearchTooLarge   = errors.New("jaeger answer exceeds the read bound")
	errTraceSearchUnreadable = errors.New("jaeger answer is not a trace search result")
	errTraceSearchNotJSON    = errors.New("jaeger answer is not JSON")
)

func searchJaegerTraces(ctx context.Context, options traceSummariesOptions, endpoint *url.URL, search traceSearch) ([]TraceSummary, error) {
	target := *endpoint
	target.RawQuery = search.jaegerQuery(options.now()).Encode()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target.String(), nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Accept", "application/json")
	response, err := options.client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, jaegerStatusFailure(response)
	}
	if !strings.Contains(response.Header.Get("Content-Type"), "json") {
		return nil, errTraceSearchNotJSON
	}
	return readJaegerSearch(&boundedReader{reader: response.Body, remaining: options.maxBytes}, search.limit)
}

func classifyTraceSearchFailure(ctx context.Context, err error, options traceSummariesOptions) *traceSearchError {
	var failure *traceSearchError
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		return &traceSearchError{
			http.StatusGatewayTimeout, "jaeger_timeout",
			fmt.Sprintf("Jaeger did not finish the trace search within %s. Show fewer traces, pick a shorter time range, or set a minimum duration.", options.timeout),
		}
	case errors.As(err, &failure):
		return failure
	case errors.Is(err, errTraceSearchTooLarge):
		return &traceSearchError{
			http.StatusBadGateway, "jaeger_response_too_large",
			fmt.Sprintf("Jaeger's answer to the trace search is larger than %d MiB. Show fewer traces or pick a shorter time range.", options.maxBytes>>20),
		}
	case errors.Is(err, errTraceSearchNotJSON):
		return &traceSearchError{
			http.StatusBadGateway, "jaeger_bad_response",
			"Jaeger answered the trace search with a page instead of trace data. Check that TARGET_JAEGER_URL points at the Jaeger query service.",
		}
	case errors.Is(err, errTraceSearchUnreadable):
		return &traceSearchError{
			http.StatusBadGateway, "jaeger_bad_response",
			"Jaeger answered the trace search with data the Dashboard could not read.",
		}
	default:
		return &traceSearchError{
			http.StatusBadGateway, "jaeger_unreachable",
			"Could not reach Jaeger at TARGET_JAEGER_URL, or the connection dropped before its answer arrived. Check that Jaeger is running and reachable from the Dashboard.",
		}
	}
}

// jaegerStatusFailure explains a non-2xx answer with the messages Jaeger puts
// in its {"errors": [{"code", "msg"}]} body.
func jaegerStatusFailure(response *http.Response) *traceSearchError {
	var body struct {
		Errors []jaegerSearchError `json:"errors"`
	}
	raw, _ := io.ReadAll(io.LimitReader(response.Body, maxJaegerErrorBodyBytes))
	_ = json.Unmarshal(raw, &body)
	detail := jaegerErrorMessages(body.Errors)
	switch response.StatusCode {
	case http.StatusBadRequest:
		message := "Jaeger rejected the trace search."
		if detail != "" {
			message = "Jaeger rejected the trace search: " + detail
		}
		return invalidTraceSearch(message)
	case http.StatusGatewayTimeout:
		return &traceSearchError{http.StatusGatewayTimeout, "jaeger_timeout", "Jaeger timed out running the trace search. Show fewer traces or pick a shorter time range."}
	default:
		message := fmt.Sprintf("Jaeger could not run the trace search (HTTP %d).", response.StatusCode)
		if detail != "" {
			message = fmt.Sprintf("Jaeger could not run the trace search (HTTP %d): %s", response.StatusCode, detail)
		}
		return &traceSearchError{http.StatusBadGateway, "jaeger_error", message}
	}
}

type jaegerSearchError struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
}

func jaegerErrorMessages(errs []jaegerSearchError) string {
	messages := make([]string, 0, len(errs))
	for _, entry := range errs {
		if message := strings.TrimSpace(entry.Msg); message != "" {
			messages = append(messages, message)
		}
	}
	joined := strings.Join(messages, "; ")
	if len(joined) > maxJaegerErrorMessage {
		joined = strings.ToValidUTF8(joined[:maxJaegerErrorMessage], "") + "…"
	}
	return joined
}

// boundedReader fails a read past its bound instead of ending the stream, so
// an oversized answer is not mistaken for a truncated one.
type boundedReader struct {
	reader    io.Reader
	remaining int64
}

func (b *boundedReader) Read(p []byte) (int, error) {
	if b.remaining <= 0 {
		return 0, errTraceSearchTooLarge
	}
	if int64(len(p)) > b.remaining {
		p = p[:b.remaining]
	}
	n, err := b.reader.Read(p)
	b.remaining -= int64(n)
	return n, err
}

// readJaegerSearch streams a search answer, {"data": [trace, ...], "errors":
// [...]}, one trace at a time, so memory holds the largest trace rather than
// the whole answer.
func readJaegerSearch(body io.Reader, limit int) ([]TraceSummary, error) {
	decoder := json.NewDecoder(body)
	if err := expectJSONDelim(decoder, '{'); err != nil {
		return nil, err
	}
	summaries := []TraceSummary{}
	positions := map[string]int{}
	var jaegerErrors []jaegerSearchError
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			return nil, unreadableJaegerSearch(err)
		}
		switch key {
		case "data":
			err = readJaegerTraces(decoder, func(summary TraceSummary) {
				// A storage backend can return one trace twice; keep the fuller copy.
				if position, seen := positions[summary.TraceID]; seen {
					if summary.SpanCount > summaries[position].SpanCount {
						summaries[position] = summary
					}
					return
				}
				positions[summary.TraceID] = len(summaries)
				summaries = append(summaries, summary)
			})
		case "errors":
			err = decodeLenient(decoder, &jaegerErrors)
		default:
			var skipped json.RawMessage
			err = decoder.Decode(&skipped)
		}
		if err != nil {
			return nil, unreadableJaegerSearch(err)
		}
	}
	if err := expectJSONDelim(decoder, '}'); err != nil {
		return nil, err
	}
	if len(summaries) == 0 && len(jaegerErrors) > 0 {
		return nil, &traceSearchError{
			http.StatusBadGateway, "jaeger_error",
			"Jaeger could not run the trace search: " + jaegerErrorMessages(jaegerErrors),
		}
	}
	slices.SortStableFunc(summaries, func(left, right TraceSummary) int {
		return cmp.Or(cmp.Compare(right.StartTimeMicros, left.StartTimeMicros), cmp.Compare(left.TraceID, right.TraceID))
	})
	if len(summaries) > limit {
		summaries = summaries[:limit]
	}
	return summaries, nil
}

func readJaegerTraces(decoder *json.Decoder, add func(TraceSummary)) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	if token == nil {
		return nil
	}
	if delim, ok := token.(json.Delim); !ok || delim != '[' {
		return fmt.Errorf("%w: data is not a list", errTraceSearchUnreadable)
	}
	for decoder.More() {
		var trace jaegerTrace
		var typeErr *json.UnmarshalTypeError
		err = decoder.Decode(&trace)
		if errors.As(err, &typeErr) {
			// The decoder has consumed the whole value, so a trace of the
			// wrong shape is left out and the stream stays readable.
			continue
		}
		if err != nil {
			return err
		}
		if summary, ok := summarizeJaegerTrace(trace); ok {
			add(summary)
		}
	}
	_, err = decoder.Token()
	return err
}

func decodeLenient(decoder *json.Decoder, value any) error {
	var typeErr *json.UnmarshalTypeError
	if err := decoder.Decode(value); err != nil && !errors.As(err, &typeErr) {
		return err
	}
	return nil
}

func expectJSONDelim(decoder *json.Decoder, want json.Delim) error {
	token, err := decoder.Token()
	if err != nil {
		return unreadableJaegerSearch(err)
	}
	if delim, ok := token.(json.Delim); !ok || delim != want {
		return fmt.Errorf("%w: expected %v", errTraceSearchUnreadable, want)
	}
	return nil
}

// unreadableJaegerSearch marks JSON that is malformed or ends early; transport
// failures and the read bound keep their own errors.
func unreadableJaegerSearch(err error) error {
	var syntaxErr *json.SyntaxError
	if errors.As(err, &syntaxErr) || errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
		return fmt.Errorf("%w: %w", errTraceSearchUnreadable, err)
	}
	return err
}

// jaegerTrace holds only the fields a summary needs; the decoder skips span
// logs and tag payloads without copying them.
type jaegerTrace struct {
	TraceID   string                   `json:"traceID"`
	Spans     []jaegerSpan             `json:"spans"`
	Processes map[string]jaegerProcess `json:"processes"`
}

type jaegerProcess struct {
	ServiceName string `json:"serviceName"`
}

type jaegerSpan struct {
	SpanID        string            `json:"spanID"`
	OperationName string            `json:"operationName"`
	References    []jaegerReference `json:"references"`
	// StartTime and Duration are microseconds; a span without them is malformed.
	StartTime *int64      `json:"startTime"`
	Duration  *int64      `json:"duration"`
	ProcessID string      `json:"processID"`
	Tags      []jaegerTag `json:"tags"`
}

type jaegerReference struct {
	RefType string `json:"refType"`
	SpanID  string `json:"spanID"`
}

type jaegerTag struct {
	Key   string         `json:"key"`
	Value jaegerTagValue `json:"value"`
}

// jaegerTagValue keeps a tag value only when it is a boolean or a short
// string, the forms an error status takes, so payload-sized values are never copied.
type jaegerTagValue string

func (value *jaegerTagValue) UnmarshalJSON(raw []byte) error {
	if len(raw) > 32 {
		return nil
	}
	if string(raw) == "true" || string(raw) == "false" {
		*value = jaegerTagValue(raw)
		return nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		*value = jaegerTagValue(text)
	}
	return nil
}

// failed reports the error marks OpenTelemetry exporters set: error=true and
// otel.status_code=ERROR.
func (span jaegerSpan) failed() bool {
	for _, tag := range span.Tags {
		if (tag.Key == "error" && tag.Value == "true") ||
			(tag.Key == "otel.status_code" && strings.EqualFold(string(tag.Value), "ERROR")) {
			return true
		}
	}
	return false
}

// hasParentIn reports whether span hangs under another span of the trace, by
// a CHILD_OF or FOLLOWS_FROM reference, as the trace waterfall draws it.
func (span jaegerSpan) hasParentIn(spanIDs map[string]struct{}) bool {
	for _, reference := range span.References {
		if reference.SpanID == span.SpanID || (reference.RefType != "CHILD_OF" && reference.RefType != "FOLLOWS_FROM") {
			continue
		}
		if _, found := spanIDs[reference.SpanID]; found {
			return true
		}
	}
	return false
}

func summarizeJaegerTrace(trace jaegerTrace) (TraceSummary, bool) {
	spans := slices.DeleteFunc(trace.Spans, func(span jaegerSpan) bool {
		return span.SpanID == "" || span.StartTime == nil || span.Duration == nil
	})
	if trace.TraceID == "" || len(spans) == 0 {
		return TraceSummary{}, false
	}
	spanIDs := make(map[string]struct{}, len(spans))
	for _, span := range spans {
		spanIDs[span.SpanID] = struct{}{}
	}

	summary := TraceSummary{TraceID: trace.TraceID, SpanCount: len(spans)}
	spanCounts := map[string]int{}
	first, root := -1, -1
	start, end := *spans[0].StartTime, *spans[0].StartTime
	for index, span := range spans {
		spanStart := *span.StartTime
		start = min(start, spanStart)
		end = max(end, spanStart+max(0, *span.Duration))
		spanCounts[trace.Processes[span.ProcessID].ServiceName]++
		if span.failed() {
			summary.ErrorCount++
		}
		if first < 0 || spanStart < *spans[first].StartTime {
			first = index
		}
		if !span.hasParentIn(spanIDs) && (root < 0 || spanStart < *spans[root].StartTime) {
			root = index
		}
	}
	if root < 0 {
		root = first
	}
	summary.RootOperation = spans[root].OperationName
	summary.RootService = trace.Processes[spans[root].ProcessID].ServiceName
	summary.StartTimeMicros = start
	summary.DurationMicros = end - start

	summary.Services = make([]TraceServiceSummary, 0, len(spanCounts))
	for name, count := range spanCounts {
		summary.Services = append(summary.Services, TraceServiceSummary{Name: name, SpanCount: count})
	}
	rank := func(service TraceServiceSummary) int {
		if service.Name == summary.RootService {
			return 0
		}
		return 1
	}
	slices.SortFunc(summary.Services, func(left, right TraceServiceSummary) int {
		return cmp.Or(
			cmp.Compare(rank(left), rank(right)),
			cmp.Compare(right.SpanCount, left.SpanCount),
			cmp.Compare(left.Name, right.Name),
		)
	})
	return summary, true
}
