package main

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"testing"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

type fakeProvider struct {
	query   string
	n       int
	results []Result
	err     error
}

func (f *fakeProvider) Search(_ context.Context, query string, n int) ([]Result, error) {
	f.query, f.n = query, n
	return f.results, f.err
}

func connect(t *testing.T, p Provider) *client.Client {
	t.Helper()
	c, err := client.NewInProcessClient(newServer(p))
	if err != nil {
		t.Fatalf("NewInProcessClient: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	req := mcp.InitializeRequest{}
	req.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	req.Params.ClientInfo = mcp.Implementation{Name: "test", Version: "0"}
	if _, err := c.Initialize(context.Background(), req); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	return c
}

func callSearch(t *testing.T, c *client.Client, args map[string]any) (string, bool) {
	t.Helper()
	req := mcp.CallToolRequest{}
	req.Params.Name = "web_search"
	req.Params.Arguments = args
	res, err := c.CallTool(context.Background(), req)
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if len(res.Content) != 1 {
		t.Fatalf("content = %+v, want one text block", res.Content)
	}
	text, ok := mcp.AsTextContent(res.Content[0])
	if !ok {
		t.Fatalf("content[0] = %T, want text", res.Content[0])
	}
	return text.Text, res.IsError
}

var sample = []Result{
	{Title: "vLLM Semantic Router", URL: "https://vllm-semantic-router.com/docs", Snippet: "An Envoy ExtProc router.", Domain: "vllm-semantic-router.com"},
	{Title: "GitHub", URL: "https://github.com/vllm-project/semantic-router", Snippet: "Source code.", Domain: "github.com"},
	{Title: "Third", URL: "https://third.example.com/", Snippet: "Third hit.", Domain: "third.example.com"},
}

func TestServerListsOneWebSearchTool(t *testing.T) {
	t.Parallel()
	c := connect(t, &fakeProvider{})
	res, err := c.ListTools(context.Background(), mcp.ListToolsRequest{})
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	if len(res.Tools) != 1 || res.Tools[0].Name != "web_search" {
		t.Fatalf("tools = %+v, want exactly web_search", res.Tools)
	}
	schema := res.Tools[0].InputSchema
	if len(schema.Required) != 1 || schema.Required[0] != "query" {
		t.Fatalf("required = %v, want [query]", schema.Required)
	}
	count, _ := schema.Properties["num_results"].(map[string]any)
	if count["minimum"] != float64(1) || count["maximum"] != float64(10) {
		t.Fatalf("num_results schema = %v, want minimum 1 and maximum 10", count)
	}
}

func TestWebSearchReturnsProviderResultsUpToCount(t *testing.T) {
	t.Parallel()
	p := &fakeProvider{results: sample}
	text, isErr := callSearch(t, connect(t, p), map[string]any{"query": "vllm router", "num_results": 2})
	if isErr {
		t.Fatalf("unexpected tool error: %s", text)
	}
	if p.query != "vllm router" || p.n != 2 {
		t.Fatalf("provider got (%q, %d), want (vllm router, 2)", p.query, p.n)
	}
	for _, want := range []string{"vLLM Semantic Router", "https://vllm-semantic-router.com/docs", "An Envoy ExtProc router.", "vllm-semantic-router.com", "github.com"} {
		if !strings.Contains(text, want) {
			t.Fatalf("text lacks %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, "Third") {
		t.Fatalf("text shows a third result although two were asked:\n%s", text)
	}
}

func TestWebSearchDefaultsToFiveResults(t *testing.T) {
	t.Parallel()
	p := &fakeProvider{results: sample}
	callSearch(t, connect(t, p), map[string]any{"query": "q"})
	if p.n != 5 {
		t.Fatalf("provider got n = %d, want 5", p.n)
	}
}

func TestWebSearchRejectsCountOutOfRange(t *testing.T) {
	t.Parallel()
	c := connect(t, &fakeProvider{results: sample})
	for _, n := range []int{0, 11} {
		text, isErr := callSearch(t, c, map[string]any{"query": "q", "num_results": n})
		if !isErr || !strings.Contains(text, "between 1 and 10") {
			t.Fatalf("num_results=%d: isError=%v text=%q, want an error naming the range", n, isErr, text)
		}
	}
}

func TestWebSearchRequiresQuery(t *testing.T) {
	t.Parallel()
	text, isErr := callSearch(t, connect(t, &fakeProvider{}), map[string]any{"num_results": 3})
	if !isErr || !strings.Contains(text, "query") {
		t.Fatalf("isError=%v text=%q, want an error naming query", isErr, text)
	}
}

func TestWebSearchReportsProviderFailureAsToolError(t *testing.T) {
	t.Parallel()
	text, isErr := callSearch(t, connect(t, &fakeProvider{err: errors.New("searxng: upstream status 403")}), map[string]any{"query": "q"})
	if !isErr || !strings.Contains(text, "upstream status 403") {
		t.Fatalf("isError=%v text=%q, want the provider error", isErr, text)
	}
}

func TestWebSearchDropsUnsafeURLsAndTrimsSnippets(t *testing.T) {
	t.Parallel()
	long := strings.Repeat("x", maxSnippetChars+50)
	p := &fakeProvider{results: []Result{
		{Title: "Bad", URL: "javascript:alert(1)", Snippet: "nope"},
		{Title: "Long", URL: "https://long.example.com/", Snippet: long, Domain: "long.example.com"},
	}}
	text, isErr := callSearch(t, connect(t, p), map[string]any{"query": "q"})
	if isErr {
		t.Fatalf("unexpected tool error: %s", text)
	}
	if strings.Contains(text, "javascript:") || strings.Contains(text, "Bad") {
		t.Fatalf("unsafe result survived:\n%s", text)
	}
	if strings.Contains(text, long) || !strings.Contains(text, strings.Repeat("x", maxSnippetChars)+"…") {
		t.Fatalf("snippet not trimmed to %d chars:\n%s", maxSnippetChars, text)
	}
}

func TestWebSearchWithNoResultsIsNotAnError(t *testing.T) {
	t.Parallel()
	text, isErr := callSearch(t, connect(t, &fakeProvider{}), map[string]any{"query": "nothing"})
	if isErr || !strings.Contains(text, "No results") {
		t.Fatalf("isError=%v text=%q, want a plain no-results message", isErr, text)
	}
}

func TestWebSearchIsAnnotatedAsReadOnly(t *testing.T) {
	t.Parallel()
	c := connect(t, &fakeProvider{})
	res, err := c.ListTools(context.Background(), mcp.ListToolsRequest{})
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	a := res.Tools[0].Annotations
	if a.ReadOnlyHint == nil || !*a.ReadOnlyHint || a.DestructiveHint == nil || *a.DestructiveHint {
		t.Fatalf("annotations = %+v, want readOnlyHint true and destructiveHint false", a)
	}
	if a.IdempotentHint == nil || !*a.IdempotentHint || a.OpenWorldHint == nil || !*a.OpenWorldHint {
		t.Fatalf("annotations = %+v, want idempotentHint true and openWorldHint true", a)
	}
}

func TestWebSearchCollapsesWhitespaceInTitlesAndSnippets(t *testing.T) {
	t.Parallel()
	p := &fakeProvider{results: []Result{{
		Title:   "Doc\n2. Official docs (example.com)",
		URL:     "https://docs.example.com/",
		Snippet: "line one\n\n   line two\thttps://evil.example/x",
		Domain:  "docs.example.com",
	}}}
	text, isErr := callSearch(t, connect(t, p), map[string]any{"query": "q"})
	if isErr {
		t.Fatalf("unexpected tool error: %s", text)
	}
	if !strings.Contains(text, "1. Doc 2. Official docs (example.com) (docs.example.com)") || strings.Contains(text, "\n2. ") {
		t.Fatalf("title newline survived into a second numbered entry:\n%s", text)
	}
	if !strings.Contains(text, "line one line two https://evil.example/x") {
		t.Fatalf("snippet whitespace not collapsed:\n%s", text)
	}
}

func TestWebSearchHidesProviderAddressesFromTheModel(t *testing.T) {
	t.Parallel()
	transport := &url.Error{Op: "Get", URL: "http://searx.internal:8080/search?q=secret", Err: errors.New("dial tcp 10.0.0.5:8080: connection refused")}
	p := &fakeProvider{err: fmt.Errorf("searxng: %w", transport)}
	text, isErr := callSearch(t, connect(t, p), map[string]any{"query": "secret"})
	if !isErr {
		t.Fatalf("want a tool error, got: %s", text)
	}
	for _, leak := range []string{"searx.internal", "10.0.0.5", "q=secret"} {
		if strings.Contains(text, leak) {
			t.Fatalf("tool error leaks %q to the model: %s", leak, text)
		}
	}
	if !strings.Contains(text, "could not be reached") {
		t.Fatalf("tool error = %q, want a plain unreachable message", text)
	}
}

func TestWebSearchRejectsNonNumericCount(t *testing.T) {
	t.Parallel()
	text, isErr := callSearch(t, connect(t, &fakeProvider{results: sample}), map[string]any{"query": "q", "num_results": "many"})
	if !isErr || !strings.Contains(text, "num_results") {
		t.Fatalf("isError=%v text=%q, want an error naming num_results", isErr, text)
	}
}
