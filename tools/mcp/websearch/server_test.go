package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

func TestStreamableHTTPEndpointServesWebSearch(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(newHTTPHandler(&fakeProvider{results: sample}))
	defer srv.Close()

	c, err := client.NewStreamableHttpClient(srv.URL + "/mcp")
	if err != nil {
		t.Fatalf("NewStreamableHttpClient: %v", err)
	}
	defer c.Close()
	ctx := context.Background()
	if err = c.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	init := mcp.InitializeRequest{}
	init.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	init.Params.ClientInfo = mcp.Implementation{Name: "test", Version: "0"}
	if _, err = c.Initialize(ctx, init); err != nil {
		t.Fatalf("Initialize over HTTP: %v", err)
	}
	tools, err := c.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		t.Fatalf("ListTools over HTTP: %v", err)
	}
	if len(tools.Tools) != 1 || tools.Tools[0].Name != "web_search" {
		t.Fatalf("tools = %+v, want exactly web_search", tools.Tools)
	}
	text, isErr := callSearch(t, c, map[string]any{"query": "vllm router", "num_results": 1})
	if isErr || text == "" {
		t.Fatalf("CallTool over HTTP: isError=%v text=%q", isErr, text)
	}
}

func TestOnlyLoopbackOriginsMayCallTheServer(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(newHTTPHandler(&fakeProvider{}))
	defer srv.Close()
	cases := []struct {
		origin string
		want   int
	}{
		{"http://attacker.example", http.StatusForbidden},
		{"http://127.0.0.1.attacker.example:8710", http.StatusForbidden},
		{"http://[::1", http.StatusForbidden},
		{"http://localhost:8710", http.StatusOK},
		{"http://127.0.0.1:8710", http.StatusOK},
		{"http://[::1]:8710", http.StatusOK},
		{"", http.StatusOK},
	}
	for _, tc := range cases {
		req, err := http.NewRequest(http.MethodPost, srv.URL+"/mcp", strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/list"}`))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json, text/event-stream")
		if tc.origin != "" {
			req.Header.Set("Origin", tc.origin)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("Origin %q: %v", tc.origin, err)
		}
		_ = resp.Body.Close()
		if resp.StatusCode != tc.want {
			t.Fatalf("Origin %q: status %d, want %d", tc.origin, resp.StatusCode, tc.want)
		}
	}
}
