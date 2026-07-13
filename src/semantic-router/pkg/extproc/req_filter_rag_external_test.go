package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildCustomRequestMarshalsTypedSubstitutions(t *testing.T) {
	router, ragConfig := customRequestTestFixture()
	query := "quote: \"; slash: \\; line:\n; unicode: 雪; markers: ${top_k} {{.Query}}"
	template := `{
		"query": "${user_content}",
		"legacy_query": "{{.Query}}",
		"top_k": ${top_k},
		"legacy_top_k": "{{.TopK}}",
		"threshold": {{.Threshold}},
		"embedded": "prefix:${user_content}:suffix"
	}`

	body := mustBuildCustomRequest(t, router, ragConfig, query, template)
	got := decodeNumberMap(t, body)

	if got["query"] != query || got["legacy_query"] != query {
		t.Fatalf("query substitutions changed user content: %#v", got)
	}
	if got["embedded"] != "prefix:"+query+":suffix" {
		t.Fatalf("embedded query substitution = %#v", got["embedded"])
	}
	if got["top_k"] != json.Number("7") || got["legacy_top_k"] != json.Number("7") {
		t.Fatalf("top_k substitutions are not typed numbers: %#v", got)
	}
	if got["threshold"] != json.Number("0.625") {
		t.Fatalf("threshold substitution = %#v, want 0.625", got["threshold"])
	}
}

func TestBuildCustomRequestPreservesConfiguredJSONNumbers(t *testing.T) {
	router, ragConfig := customRequestTestFixture()
	template := `{
		"query":"${user_content}",
		"max_id":9223372036854775807,
		"precise":0.12345678901234567890123456789
	}`

	body := mustBuildCustomRequest(t, router, ragConfig, "query", template)
	got := decodeNumberMap(t, body)

	if got["max_id"] != json.Number("9223372036854775807") {
		t.Fatalf("max_id = %#v, want exact MaxInt64 JSON number", got["max_id"])
	}
	if got["precise"] != json.Number("0.12345678901234567890123456789") {
		t.Fatalf("precise = %#v, want original JSON number", got["precise"])
	}
}

func TestBuildCustomRequestKeepsConfiguredFieldBoundary(t *testing.T) {
	router, ragConfig := customRequestTestFixture()
	query := `value","admin":true,"other":"owned`
	body := mustBuildCustomRequest(
		t,
		router,
		ragConfig,
		query,
		`{"query":"${user_content}","fixed":true}`,
	)
	got := decodeNumberMap(t, body)

	if len(got) != 2 || got["query"] != query || got["fixed"] != true {
		t.Fatalf("user content changed the configured object structure: %#v", got)
	}
	if _, ok := got["admin"]; ok {
		t.Fatalf("user content injected an object field: %#v", got)
	}
}

func TestBuildCustomRequestRejectsInvalidDocuments(t *testing.T) {
	router, ragConfig := customRequestTestFixture()
	tests := []struct {
		name     string
		template string
		want     string
	}{
		{
			name:     "incomplete document",
			template: `{"query":"${user_content}"`,
			want:     "invalid custom request template JSON",
		},
		{
			name:     "multiple documents",
			template: `{"query":"${user_content}"} {"second":true}`,
			want:     "template contains multiple JSON values",
		},
		{
			name:     "trailing non-whitespace",
			template: `{"query":"${user_content}"} trailing`,
			want:     "invalid trailing data",
		},
		{
			name:     "placeholder in object key",
			template: `{"${user_content}":"value"}`,
			want:     "not allowed in object keys",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := router.buildCustomRequest(
				&RequestContext{UserContent: "query"},
				ragConfig,
				test.template,
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("buildCustomRequest() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestExternalRAGEndToEnd(t *testing.T) {
	query := "quoted \"value\" with \\ slash and 雪 ${top_k}"
	requestBody := make(chan []byte, 1)
	responseBody := []byte(`{"content":"retrieved external context"}`)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			requestBody <- []byte("read error: " + err.Error())
			return
		}
		requestBody <- body
		_, _ = w.Write(responseBody)
	}))
	defer server.Close()

	limit := int64(len(responseBody))
	got, err := retrieveExternalRAG(context.Background(), server.URL, &limit, query)
	if err != nil {
		t.Fatalf("retrieveFromExternalAPI() error = %v", err)
	}
	if got != "retrieved external context" {
		t.Fatalf("retrieveFromExternalAPI() = %q", got)
	}

	outbound := decodeNumberMap(t, <-requestBody)
	if outbound["query"] != query {
		t.Fatalf("outbound query = %#v, want exact user content", outbound["query"])
	}
}

func TestExternalRAGAcceptsResponseAtExactLimit(t *testing.T) {
	body := []byte(`{"content":"exact"}`)
	server := fixedResponseServer(body)
	defer server.Close()

	limit := int64(len(body))
	got, err := retrieveExternalRAG(context.Background(), server.URL, &limit, "query")
	if err != nil {
		t.Fatalf("retrieveFromExternalAPI() error = %v", err)
	}
	if got != "exact" {
		t.Fatalf("retrieveFromExternalAPI() = %q, want exact", got)
	}
}

func TestExternalRAGRejectsResponseOneByteOverLimit(t *testing.T) {
	validPrefix := []byte(`{"content":"prefix"}`)
	server := fixedResponseServer(append(validPrefix, ' '))
	defer server.Close()

	limit := int64(len(validPrefix))
	_, err := retrieveExternalRAG(context.Background(), server.URL, &limit, "query")
	want := fmt.Sprintf("exceeded configured limit of %d bytes", len(validPrefix))
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("retrieveFromExternalAPI() error = %v, want %q", err, want)
	}
}

func TestExternalRAGAcceptsBoundedChunkedResponse(t *testing.T) {
	chunks := []string{`{"con`, `tent":"`, `chunked"}`}
	bodyLength := len(strings.Join(chunks, ""))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		flusher := w.(http.Flusher)
		for _, chunk := range chunks {
			_, _ = w.Write([]byte(chunk))
			flusher.Flush()
		}
	}))
	defer server.Close()

	limit := int64(bodyLength)
	got, err := retrieveExternalRAG(context.Background(), server.URL, &limit, "query")
	if err != nil {
		t.Fatalf("retrieveFromExternalAPI() error = %v", err)
	}
	if got != "chunked" {
		t.Fatalf("retrieveFromExternalAPI() = %q, want chunked", got)
	}
}

func TestExternalRAGPropagatesCancellationWhileReading(t *testing.T) {
	started := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		_, _ = w.Write([]byte(`{"content":"`))
		w.(http.Flusher).Flush()
		close(started)
		<-request.Context().Done()
	}))
	defer server.Close()

	requestContext, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	limit := int64(1024)
	go func() {
		_, err := retrieveExternalRAG(requestContext, server.URL, &limit, "query")
		result <- err
	}()

	waitForServerAndCancel(t, started, cancel)
	err := waitForRetrievalResult(t, result)
	if err == nil || !strings.Contains(err.Error(), "context canceled") {
		t.Fatalf("retrieveFromExternalAPI() error = %v, want context canceled", err)
	}
}

func TestExternalRAGResponseLimitBoundaries(t *testing.T) {
	if got := externalAPIResponseBodyLimit(&config.ExternalAPIRAGConfig{}); got != defaultExternalAPIMaxResponseBodyBytes {
		t.Fatalf("default response limit = %d, want %d", got, defaultExternalAPIMaxResponseBodyBytes)
	}

	limit := int64(math.MaxInt64)
	if got := externalAPIResponseBodyLimit(&config.ExternalAPIRAGConfig{MaxResponseBodyBytes: &limit}); got != math.MaxInt64 {
		t.Fatalf("configured response limit = %d, want MaxInt64", got)
	}

	body, exceeded, err := readExternalAPIResponseBody(strings.NewReader("bounded"), math.MaxInt64)
	if err != nil || exceeded || string(body) != "bounded" {
		t.Fatalf("readExternalAPIResponseBody(MaxInt64) = (%q, %t, %v)", body, exceeded, err)
	}
}

func customRequestTestFixture() (*OpenAIRouter, *config.RAGPluginConfig) {
	topK := 7
	threshold := float32(0.625)
	return &OpenAIRouter{}, &config.RAGPluginConfig{
		TopK:                &topK,
		SimilarityThreshold: &threshold,
	}
}

func mustBuildCustomRequest(
	t *testing.T,
	router *OpenAIRouter,
	ragConfig *config.RAGPluginConfig,
	query string,
	template string,
) []byte {
	t.Helper()
	body, err := router.buildCustomRequest(&RequestContext{UserContent: query}, ragConfig, template)
	if err != nil {
		t.Fatalf("buildCustomRequest() error = %v", err)
	}
	return body
}

func decodeNumberMap(t *testing.T, data []byte) map[string]interface{} {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var got map[string]interface{}
	if err := decoder.Decode(&got); err != nil {
		t.Fatalf("decode JSON object: %v\n%s", err, data)
	}
	return got
}

func retrieveExternalRAG(
	requestContext context.Context,
	serverURL string,
	limit *int64,
	query string,
) (string, error) {
	router := &OpenAIRouter{}
	ragConfig := &config.RAGPluginConfig{
		Enabled: true,
		Backend: "external_api",
		BackendConfig: config.MustStructuredPayload(&config.ExternalAPIRAGConfig{
			Endpoint:             serverURL,
			RequestFormat:        "custom",
			RequestTemplate:      `{"query":"${user_content}"}`,
			MaxResponseBodyBytes: limit,
		}),
	}
	return router.retrieveFromExternalAPI(requestContext, &RequestContext{UserContent: query}, ragConfig)
}

func fixedResponseServer(body []byte) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
}

func waitForServerAndCancel(t *testing.T, started <-chan struct{}, cancel context.CancelFunc) {
	t.Helper()
	select {
	case <-started:
		cancel()
	case <-time.After(2 * time.Second):
		cancel()
		t.Fatal("server did not start streaming")
	}
}

func waitForRetrievalResult(t *testing.T, result <-chan error) error {
	t.Helper()
	select {
	case err := <-result:
		return err
	case <-time.After(2 * time.Second):
		t.Fatal("retrieveFromExternalAPI() did not return after cancellation")
		return nil
	}
}
