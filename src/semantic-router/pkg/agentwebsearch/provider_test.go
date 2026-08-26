package agentwebsearch

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentruntime"
)

const testNamespaceID = "5bd26255-2984-4d82-9fc4-06f93ba70731"

var _ agentruntime.NativeToolProvider = (*Provider)(nil)

func TestProviderSearchesThroughRegisteredRouterTool(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Query().Get("q") != "vLLM Semantic Router" {
			t.Fatalf("query = %q", request.URL.Query().Get("q"))
		}
		_, _ = response.Write([]byte(`<html><body>
<div class="result results_links">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Frouter%23old">Semantic <b>Router</b></a>
  <a class="result__snippet">A &amp; B <strong>routing</strong> result.</a>
</div>
<div class="result results_links">
  <a class="result__a" href="https://example.org/docs">Documentation</a>
  <span class="result__snippet">Current docs.</span>
</div>
</body></html>`))
	}))
	defer server.Close()

	provider, err := New(Options{Client: server.Client(), Endpoint: server.URL + "/html/"})
	if err != nil {
		t.Fatal(err)
	}
	tools, err := provider.Current(context.Background(), testNamespaceID)
	if err != nil || len(tools) != 1 {
		t.Fatalf("Current() = (%v, %v)", tools, err)
	}
	if tools[0].Definition.Name != agentmanagement.ToolWebSearch {
		t.Fatalf("tool name = %q", tools[0].Definition.Name)
	}
	handler, err := provider.Resolve(context.Background(), testNamespaceID, tools[0].Definition)
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler.Invoke(
		context.Background(), agentmanagement.ToolInvocationContext{},
		json.RawMessage(`{"query":"  vLLM   Semantic Router ","maxResults":2}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	var got searchOutput
	if err := json.Unmarshal(result.Value, &got); err != nil {
		t.Fatal(err)
	}
	want := searchOutput{
		Query: "vLLM Semantic Router",
		Results: []searchResult{
			{Title: "Semantic Router", URL: "https://example.com/router", Snippet: "A & B routing result.", Domain: "example.com"},
			{Title: "Documentation", URL: "https://example.org/docs", Snippet: "Current docs.", Domain: "example.org"},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("search output = %#v, want %#v", got, want)
	}
}

func TestProviderRejectsUnknownInputAndDefinitionDrift(t *testing.T) {
	provider, err := New(Options{Client: http.DefaultClient})
	if err != nil {
		t.Fatal(err)
	}
	tools, err := provider.Current(context.Background(), testNamespaceID)
	if err != nil {
		t.Fatal(err)
	}
	definition := tools[0].Definition
	definition.TimeoutMilliseconds++
	if _, err := provider.Resolve(context.Background(), testNamespaceID, definition); !errors.Is(err, agentmanagement.ErrConflict) {
		t.Fatalf("Resolve() drift error = %v", err)
	}
	if _, err := provider.search(
		context.Background(), agentmanagement.ToolInvocationContext{},
		json.RawMessage(`{"query":"router","unexpected":true}`),
	); !errors.Is(err, agentmanagement.ErrInvalid) {
		t.Fatalf("search() invalid input error = %v", err)
	}
}

func TestParseResultsBoundsAndFiltersURLs(t *testing.T) {
	html := `<div class="result"><a class="result__a" href="javascript:alert(1)">Bad</a></div>
<div class="result"><a class="result__a" href="https://safe.example/a">First</a></div>
<div class="result"><a class="result__a" href="https://safe.example/a">Duplicate</a></div>
<div class="result"><a class="result__a" href="https://next.example/b">Second</a></div>`
	results, err := parseResults(strings.NewReader(html), 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].URL != "https://safe.example/a" {
		t.Fatalf("results = %#v", results)
	}
}
