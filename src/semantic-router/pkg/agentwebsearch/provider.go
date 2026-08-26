// Package agentwebsearch exposes bounded public web search as a Router-native
// Agent tool. The browser never receives an upstream search credential or
// executes the tool; invocation remains inside the Agent authorization and
// transcript boundary.
package agentwebsearch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"reflect"
	"strings"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

const (
	defaultEndpoint = "https://html.duckduckgo.com/html/"
	toolTimeoutMS   = int64(15_000)
)

var (
	inputSchema = json.RawMessage(`{
  "type":"object",
  "additionalProperties":false,
  "properties":{
    "query":{"type":"string","minLength":1,"maxLength":500},
    "maxResults":{"type":"integer","minimum":1,"maximum":8}
  },
  "required":["query"]
}`)
	outputSchema = json.RawMessage(`{
  "type":"object",
  "additionalProperties":false,
  "properties":{
    "query":{"type":"string"},
    "results":{
      "type":"array",
      "maxItems":8,
      "items":{
        "type":"object",
        "additionalProperties":false,
        "properties":{
          "title":{"type":"string"},
          "url":{"type":"string"},
          "snippet":{"type":"string"},
          "domain":{"type":"string"}
        },
        "required":["title","url","snippet","domain"]
      }
    }
  },
  "required":["query","results"]
}`)
)

type Options struct {
	Client   *http.Client
	Endpoint string
}

// Provider is an immutable NativeToolProvider backed by a fixed, operator
// authorized search origin. Endpoint injection exists for deterministic tests;
// production uses the guarded default origin.
type Provider struct {
	client       *http.Client
	endpoint     *url.URL
	registration agentmanagement.RegisteredTool
}

func New(options Options) (*Provider, error) {
	if options.Client == nil {
		return nil, errors.New("Agent web search HTTP client is unavailable")
	}
	endpoint := strings.TrimSpace(options.Endpoint)
	if endpoint == "" {
		endpoint = defaultEndpoint
	}
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" || parsed.RawQuery != "" || parsed.Fragment != "" ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, errors.New("Agent web search endpoint is invalid")
	}
	definition, err := agentmanagement.CanonicalizeToolDefinition(agentmanagement.ToolDefinition{
		Name:         agentmanagement.ToolWebSearch,
		Description:  "Search the public web for current sources and concise result snippets.",
		InputSchema:  append(json.RawMessage(nil), inputSchema...),
		OutputSchema: append(json.RawMessage(nil), outputSchema...),
		RequiredPermissions: []accesscontrol.Permission{
			accesscontrol.PermissionToolRead,
		},
		Class:               agentmanagement.ToolRead,
		Idempotency:         agentmanagement.ToolInvocationIdempotent,
		TimeoutMilliseconds: toolTimeoutMS,
	})
	if err != nil {
		return nil, fmt.Errorf("define Agent web search tool: %w", err)
	}
	provider := &Provider{client: options.Client, endpoint: parsed}
	provider.registration = agentmanagement.RegisteredTool{
		Definition: definition,
		Handler:    agentmanagement.ToolHandlerFunc(provider.search),
		Origin:     agentmanagement.ToolOrigin{Kind: agentmanagement.ToolOriginRouter},
	}
	return provider, nil
}

func (provider *Provider) Current(
	_ context.Context, namespaceID string,
) ([]agentmanagement.RegisteredTool, error) {
	if provider == nil || uuid.Validate(namespaceID) != nil {
		return nil, agentmanagement.ErrInvalid
	}
	return []agentmanagement.RegisteredTool{cloneRegistration(provider.registration)}, nil
}

func (provider *Provider) Resolve(
	_ context.Context, namespaceID string, requested agentmanagement.ToolDefinition,
) (agentmanagement.ToolHandler, error) {
	if provider == nil || uuid.Validate(namespaceID) != nil {
		return nil, agentmanagement.ErrInvalid
	}
	if requested.Name != provider.registration.Definition.Name {
		return nil, agentmanagement.ErrToolUnavailable
	}
	canonical, err := agentmanagement.CanonicalizeToolDefinition(requested)
	if err != nil || !reflect.DeepEqual(canonical, provider.registration.Definition) {
		return nil, agentmanagement.ErrConflict
	}
	return provider.registration.Handler, nil
}

func cloneRegistration(source agentmanagement.RegisteredTool) agentmanagement.RegisteredTool {
	result := source
	result.Definition.InputSchema = append(json.RawMessage(nil), source.Definition.InputSchema...)
	result.Definition.OutputSchema = append(json.RawMessage(nil), source.Definition.OutputSchema...)
	result.Definition.RequiredPermissions = append(
		[]accesscontrol.Permission(nil), source.Definition.RequiredPermissions...,
	)
	return result
}

func (provider *Provider) search(
	ctx context.Context, _ agentmanagement.ToolInvocationContext, raw json.RawMessage,
) (agentmanagement.ToolResult, error) {
	input, err := decodeInput(raw)
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	requestURL := *provider.endpoint
	query := requestURL.Query()
	query.Set("q", input.Query)
	requestURL.RawQuery = query.Encode()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
	if err != nil {
		return agentmanagement.ToolResult{}, fmt.Errorf("create web search request: %w", err)
	}
	request.Header.Set("Accept", "text/html,application/xhtml+xml")
	request.Header.Set("Accept-Language", "en-US,en;q=0.8")
	request.Header.Set("User-Agent", "Mozilla/5.0 (compatible; vLLM-Semantic-Router/1.0)")
	response, err := provider.client.Do(request)
	if err != nil {
		return agentmanagement.ToolResult{}, fmt.Errorf("search the public web: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return agentmanagement.ToolResult{}, fmt.Errorf("search the public web: upstream returned HTTP %d", response.StatusCode)
	}
	results, err := parseResults(response.Body, input.MaxResults)
	if err != nil {
		return agentmanagement.ToolResult{}, fmt.Errorf("read public web results: %w", err)
	}
	encoded, err := json.Marshal(searchOutput{Query: input.Query, Results: results})
	if err != nil {
		return agentmanagement.ToolResult{}, fmt.Errorf("encode public web results: %w", err)
	}
	return agentmanagement.ToolResult{Value: encoded}, nil
}

type searchInput struct {
	Query      string `json:"query"`
	MaxResults int    `json:"maxResults,omitempty"`
}

type searchOutput struct {
	Query   string         `json:"query"`
	Results []searchResult `json:"results"`
}

type searchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
	Domain  string `json:"domain"`
}

func decodeInput(raw json.RawMessage) (searchInput, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var input searchInput
	if err := decoder.Decode(&input); err != nil {
		return searchInput{}, agentmanagement.ErrInvalid
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return searchInput{}, agentmanagement.ErrInvalid
	}
	input.Query = strings.Join(strings.Fields(input.Query), " ")
	if input.MaxResults == 0 {
		input.MaxResults = 5
	}
	if input.Query == "" || len([]rune(input.Query)) > 500 || input.MaxResults < 1 || input.MaxResults > 8 {
		return searchInput{}, agentmanagement.ErrInvalid
	}
	return input, nil
}
