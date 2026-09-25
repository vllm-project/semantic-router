package mcp

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
)

type pagedListServer struct {
	*httptest.Server
	mu      sync.Mutex
	cursors map[string][]string
}

// newPagedListServer serves tools/list, resources/list, and prompts/list one
// item per page and records the cursor that each list request carried.
func newPagedListServer(t *testing.T, pages int) *pagedListServer {
	t.Helper()
	pageByCursor := map[string]int{"": 1}
	for page := 2; page <= pages; page++ {
		pageByCursor[pageCursor(page)] = page
	}

	server := &pagedListServer{cursors: map[string][]string{}}
	server.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method := strings.TrimPrefix(r.URL.Path, "/")
		if method == "initialize" {
			_, _ = w.Write([]byte(`{}`))
			return
		}

		var request struct {
			Params struct {
				Cursor string `json:"cursor"`
			} `json:"params"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		server.mu.Lock()
		server.cursors[method] = append(server.cursors[method], request.Params.Cursor)
		server.mu.Unlock()

		page, ok := pageByCursor[request.Params.Cursor]
		if !ok {
			http.Error(w, "invalid cursor", http.StatusBadRequest)
			return
		}
		next := mcp.PaginatedResult{}
		if page < pages {
			next.NextCursor = mcp.Cursor(pageCursor(page + 1))
		}

		var result any
		switch method {
		case "tools/list":
			result = mcp.ListToolsResult{
				PaginatedResult: next,
				Tools:           []mcp.Tool{mcp.NewTool(fmt.Sprintf("tool-%d", page))},
			}
		case "resources/list":
			name := fmt.Sprintf("resource-%d", page)
			result = mcp.ListResourcesResult{
				PaginatedResult: next,
				Resources:       []mcp.Resource{mcp.NewResource("file:///"+name, name)},
			}
		case "prompts/list":
			result = mcp.ListPromptsResult{
				PaginatedResult: next,
				Prompts:         []mcp.Prompt{mcp.NewPrompt(fmt.Sprintf("prompt-%d", page))},
			}
		default:
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(result)
	}))
	t.Cleanup(server.Close)
	return server
}

func (s *pagedListServer) cursorsSent(method string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.cursors[method])
}

// pageCursor produces opaque cursors like the example in the MCP pagination spec.
func pageCursor(page int) string {
	return base64.StdEncoding.EncodeToString(fmt.Appendf(nil, `{"page": %d}`, page))
}

func itemNames[T any](items []T, name func(T) string) []string {
	names := make([]string, 0, len(items))
	for _, item := range items {
		names = append(names, name(item))
	}
	return names
}

func toolNames(tools []mcp.Tool) []string {
	return itemNames(tools, func(tool mcp.Tool) string { return tool.Name })
}

func TestHTTPClientLoadsEveryListPage(t *testing.T) {
	server := newPagedListServer(t, 3)
	client := NewHTTPClient("paged", ClientConfig{URL: server.URL})
	if err := client.Connect(); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	wantCursors := []string{"", pageCursor(2), pageCursor(3)}
	for _, tc := range []struct {
		method string
		loaded []string
		want   []string
	}{
		{
			method: "tools/list",
			loaded: toolNames(client.GetTools()),
			want:   []string{"tool-1", "tool-2", "tool-3"},
		},
		{
			method: "resources/list",
			loaded: itemNames(client.GetResources(), func(resource mcp.Resource) string { return resource.Name }),
			want:   []string{"resource-1", "resource-2", "resource-3"},
		},
		{
			method: "prompts/list",
			loaded: itemNames(client.GetPrompts(), func(prompt mcp.Prompt) string { return prompt.Name }),
			want:   []string{"prompt-1", "prompt-2", "prompt-3"},
		},
	} {
		t.Run(tc.method, func(t *testing.T) {
			if !slices.Equal(tc.loaded, tc.want) {
				t.Errorf("loaded %v, want %v", tc.loaded, tc.want)
			}
			if got := server.cursorsSent(tc.method); !slices.Equal(got, wantCursors) {
				t.Errorf("cursors sent %q, want %q", got, wantCursors)
			}
		})
	}
}
