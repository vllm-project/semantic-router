package mcp

import (
	"context"
	"testing"

	mcpproto "github.com/mark3labs/mcp-go/mcp"
)

func TestCallToolStreamingReportsTerminalChunks(t *testing.T) {
	t.Parallel()

	t.Run("success preserves the full text result", func(t *testing.T) {
		manager, err := newManagerWithInProcessTool("srv", "echo", func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error) {
			return &mcpproto.CallToolResult{
				Content: []mcpproto.Content{mcpproto.TextContent{Type: "text", Text: "live-repro-ok"}},
			}, nil
		})
		if err != nil {
			t.Fatal(err)
		}

		var chunk StreamChunk
		err = manager.ExecuteToolStreaming(context.Background(), "srv", "echo", []byte(`{}`), func(got StreamChunk) error {
			chunk = got
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if chunk.Type != "complete" || chunk.Data != "live-repro-ok" || chunk.Progress != 100 {
			t.Fatalf("chunk = %#v", chunk)
		}
	})

	t.Run("multiple content items are preserved", func(t *testing.T) {
		manager, err := newManagerWithInProcessTool("srv", "echo", func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error) {
			return &mcpproto.CallToolResult{
				Content: []mcpproto.Content{
					mcpproto.TextContent{Type: "text", Text: "one"},
					mcpproto.TextContent{Type: "text", Text: "two"},
				},
			}, nil
		})
		if err != nil {
			t.Fatal(err)
		}

		var chunk StreamChunk
		if err := manager.ExecuteToolStreaming(context.Background(), "srv", "echo", []byte(`{}`), func(got StreamChunk) error {
			chunk = got
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		items, ok := chunk.Data.([]ContentItem)
		if chunk.Type != "complete" || !ok || len(items) != 2 || items[0].Text != "one" || items[1].Text != "two" {
			t.Fatalf("chunk = %#v", chunk)
		}
	})

	t.Run("tool isError is a failure chunk and keeps the payload", func(t *testing.T) {
		manager, err := newManagerWithInProcessTool("srv", "echo", func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error) {
			return &mcpproto.CallToolResult{
				IsError: true,
				Content: []mcpproto.Content{mcpproto.TextContent{Type: "text", Text: "tool failed"}},
			}, nil
		})
		if err != nil {
			t.Fatal(err)
		}

		var chunk StreamChunk
		err = manager.ExecuteToolStreaming(context.Background(), "srv", "echo", []byte(`{}`), func(got StreamChunk) error {
			chunk = got
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if chunk.Type != "error" || chunk.Data != "tool failed" {
			t.Fatalf("chunk = %#v", chunk)
		}
	})

	t.Run("transport failure returns an error and a redacted chunk", func(t *testing.T) {
		manager, err := newManagerWithInProcessTool("srv", "echo", func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error) {
			return nil, context.DeadlineExceeded
		})
		if err != nil {
			t.Fatal(err)
		}

		var chunk StreamChunk
		err = manager.ExecuteToolStreaming(context.Background(), "srv", "echo", []byte(`{}`), func(got StreamChunk) error {
			chunk = got
			return nil
		})
		if err == nil {
			t.Fatal("expected transport error")
		}
		if chunk.Type != "error" || chunk.Data != "Tool execution failed" {
			t.Fatalf("chunk = %#v err=%v", chunk, err)
		}
	})
}
