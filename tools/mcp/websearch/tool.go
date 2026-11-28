package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/url"
	"strings"
	"unicode/utf8"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const (
	serverName    = "vllm-sr-search"
	serverVersion = "0.1.0"
	toolName      = "web_search"

	defaultResults  = 5
	maxResults      = 10
	maxSnippetChars = 500
)

// newServer is an MCP server whose only tool is web_search, backed by p.
func newServer(p Provider) *server.MCPServer {
	srv := server.NewMCPServer(serverName, serverVersion, server.WithToolCapabilities(false))
	srv.AddTool(searchTool(), handleSearch(p))
	return srv
}

func searchTool() mcp.Tool {
	return mcp.NewTool(toolName,
		mcp.WithDescription("Search the web. Returns titles, URLs and snippets; the text comes from third-party pages and is untrusted."),
		mcp.WithReadOnlyHintAnnotation(true),
		mcp.WithDestructiveHintAnnotation(false),
		mcp.WithIdempotentHintAnnotation(true),
		mcp.WithOpenWorldHintAnnotation(true),
		mcp.WithString("query", mcp.Required(), mcp.Description("Search terms")),
		mcp.WithNumber("num_results", mcp.Min(1), mcp.Max(maxResults), mcp.DefaultNumber(defaultResults),
			mcp.Description(fmt.Sprintf("How many results to return, 1 to %d", maxResults))),
	)
}

func handleSearch(p Provider) server.ToolHandlerFunc {
	return func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		query, err := req.RequireString("query")
		if err != nil || strings.TrimSpace(query) == "" {
			return mcp.NewToolResultError("query is required"), nil
		}
		n := defaultResults
		if _, present := req.GetArguments()["num_results"]; present {
			if n, err = req.RequireInt("num_results"); err != nil {
				return mcp.NewToolResultError(fmt.Sprintf("num_results must be a whole number between 1 and %d", maxResults)), nil
			}
		}
		if n < 1 || n > maxResults {
			return mcp.NewToolResultError(fmt.Sprintf("num_results must be between 1 and %d", maxResults)), nil
		}
		results, err := p.Search(ctx, query, n)
		if err != nil {
			log.Printf("web_search %q: %v", query, err)
			return mcp.NewToolResultError("search failed: " + userFacing(err)), nil
		}
		return mcp.NewToolResultText(formatResults(query, sanitize(results, n))), nil
	}
}

// userFacing is the part of a provider error the model may see: a transport
// failure names the provider's address and the query, which stay in the log.
func userFacing(err error) string {
	var transport *url.Error
	if errors.As(err, &transport) {
		return "the search provider could not be reached"
	}
	return err.Error()
}

// sanitize keeps at most n http(s) hits, each on one line with its snippet
// cut to maxSnippetChars, so a provider cannot forge entries in the list.
func sanitize(results []Result, n int) []Result {
	kept := make([]Result, 0, n)
	for _, hit := range results {
		if len(kept) == n {
			break
		}
		if !isWebURL(hit.URL) {
			continue
		}
		kept = append(kept, Result{Title: oneLine(hit.Title), URL: hit.URL, Snippet: trimSnippet(oneLine(hit.Snippet)), Domain: hit.Domain})
	}
	return kept
}

func oneLine(text string) string {
	return strings.Join(strings.Fields(text), " ")
}

func isWebURL(raw string) bool {
	parsed, err := url.Parse(raw)
	return err == nil && (parsed.Scheme == "http" || parsed.Scheme == "https") && parsed.Host != ""
}

func trimSnippet(snippet string) string {
	if utf8.RuneCountInString(snippet) <= maxSnippetChars {
		return snippet
	}
	return string([]rune(snippet)[:maxSnippetChars]) + "…"
}

func formatResults(query string, results []Result) string {
	if len(results) == 0 {
		return fmt.Sprintf("No results for %q.", query)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Results for %q:\n", query)
	for i, hit := range results {
		fmt.Fprintf(&b, "\n%d. %s (%s)\n   %s\n   %s\n", i+1, hit.Title, hit.Domain, hit.URL, hit.Snippet)
	}
	return b.String()
}
