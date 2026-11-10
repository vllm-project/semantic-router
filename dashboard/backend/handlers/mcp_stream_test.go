package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	mcpproto "github.com/mark3labs/mcp-go/mcp"

	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
)

// preFixSSE rewrites fixed handler output to the old always-event-message framing.
func preFixSSE(body string) string {
	body = strings.ReplaceAll(body, "event: complete\n", "event: message\n")
	body = strings.ReplaceAll(body, "event: error\n", "event: message\n")
	body = strings.ReplaceAll(body, "event: progress\n", "event: message\n")
	body = strings.ReplaceAll(body, "event: partial\n", "event: message\n")
	return body
}

// legacyFrontendStreamResult mirrors pre-fix executeToolStreaming (#4468): chunk type = SSE event name only.
func legacyFrontendStreamResult(body string) (success bool, result interface{}, lastChunkType string) {
	lines := strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n")
	eventType := ""
	eventData := ""
	var finalResult interface{}

	flush := func() {
		if eventData == "" {
			return
		}
		lastChunkType = eventType
		if eventType == "complete" {
			var parsed interface{}
			if err := json.Unmarshal([]byte(eventData), &parsed); err == nil {
				finalResult = parsed
			}
		}
		eventType = ""
		eventData = ""
	}

	for _, line := range lines {
		if strings.HasPrefix(line, "event:") {
			eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			eventData = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		} else if line == "" && eventData != "" {
			flush()
		}
	}
	return true, finalResult, lastChunkType
}

// currentFrontendStreamResult mirrors fixed executeToolStreaming terminal semantics.
func currentFrontendStreamResult(body string) (success bool, result interface{}, errMsg string) {
	lines := strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n")
	eventType := ""
	eventData := ""
	sawComplete := false
	sawError := false
	var terminalResult interface{}
	var terminalError string

	flush := func() {
		if eventData == "" {
			return
		}
		chunkType := eventType
		var parsed map[string]interface{}
		if err := json.Unmarshal([]byte(eventData), &parsed); err != nil {
			chunkType = "partial"
		} else if t, ok := parsed["type"].(string); ok && t != "" {
			chunkType = t
		}
		switch chunkType {
		case "complete":
			sawComplete = true
			terminalResult = parsed["data"]
		case "error":
			sawError = true
			terminalResult = parsed["data"]
			if s, ok := parsed["data"].(string); ok && s != "" {
				terminalError = s
			} else {
				terminalError = "Tool execution failed"
			}
		}
		eventType = ""
		eventData = ""
	}

	for _, line := range lines {
		if strings.HasPrefix(line, "event:") {
			eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			eventData = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		} else if line == "" {
			flush()
		}
	}
	flush()

	if sawError || !sawComplete {
		if terminalError == "" {
			terminalError = "Stream ended before a complete result"
		}
		return false, terminalResult, terminalError
	}
	return true, terminalResult, ""
}

func TestFullStackMCPStreamingLiveRepro(t *testing.T) {
	scenarios := []struct {
		name       string
		call       func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error)
		wantLegacy struct {
			success   bool
			resultNil bool
			chunkType string
		}
		wantFixed struct {
			success bool
			result  interface{}
			errMsg  string
		}
	}{
		{
			name: "successful tool",
			call: func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error) {
				return &mcpproto.CallToolResult{
					Content: []mcpproto.Content{mcpproto.TextContent{Type: "text", Text: "live-repro-ok"}},
				}, nil
			},
			wantLegacy: struct {
				success   bool
				resultNil bool
				chunkType string
			}{success: true, resultNil: true, chunkType: "message"},
			wantFixed: struct {
				success bool
				result  interface{}
				errMsg  string
			}{success: true, result: "live-repro-ok"},
		},
		{
			name: "mcp isError",
			call: func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error) {
				return &mcpproto.CallToolResult{
					IsError: true,
					Content: []mcpproto.Content{mcpproto.TextContent{Type: "text", Text: "tool failed"}},
				}, nil
			},
			wantLegacy: struct {
				success   bool
				resultNil bool
				chunkType string
			}{success: true, resultNil: true, chunkType: "message"},
			wantFixed: struct {
				success bool
				result  interface{}
				errMsg  string
			}{success: false, result: "tool failed", errMsg: "tool failed"},
		},
		{
			name: "transport failure",
			call: func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error) {
				return nil, errors.New("secret upstream boom")
			},
			wantLegacy: struct {
				success   bool
				resultNil bool
				chunkType string
			}{success: true, resultNil: true, chunkType: "message"},
			wantFixed: struct {
				success bool
				result  interface{}
				errMsg  string
			}{success: false, result: "Tool execution failed", errMsg: "Tool execution failed"},
		},
	}

	for _, sc := range scenarios {
		t.Run(sc.name, func(t *testing.T) {
			manager, err := mcp.NewManagerWithInProcessTool("srv", "echo", sc.call)
			if err != nil {
				t.Fatal(err)
			}
			sse := postToolStream(t, manager, `{"server_id":"srv","tool_name":"echo","arguments":{}}`)

			oldSSE := preFixSSE(sse)
			legacyOK, legacyResult, legacyType := legacyFrontendStreamResult(oldSSE)
			fixedOK, fixedResult, fixedErr := currentFrontendStreamResult(sse)

			t.Logf("--- %s ---", sc.name)
			t.Logf("fixed handler SSE:\n%s", sse)
			t.Logf("pre-fix SSE (event: message only):\n%s", oldSSE)
			t.Logf("legacy parser on pre-fix SSE: success=%v result=%v lastChunkType=%q", legacyOK, legacyResult, legacyType)
			t.Logf("fixed parser:  success=%v result=%v error=%q", fixedOK, fixedResult, fixedErr)

			if legacyType != sc.wantLegacy.chunkType {
				t.Fatalf("legacy chunk type = %q want %q", legacyType, sc.wantLegacy.chunkType)
			}
			if legacyOK != sc.wantLegacy.success {
				t.Fatalf("legacy success = %v want %v", legacyOK, sc.wantLegacy.success)
			}
			if sc.wantLegacy.resultNil && legacyResult != nil {
				t.Fatalf("legacy result = %v want nil", legacyResult)
			}

			if fixedOK != sc.wantFixed.success {
				t.Fatalf("fixed success = %v want %v", fixedOK, sc.wantFixed.success)
			}
			if fmt.Sprint(fixedResult) != fmt.Sprint(sc.wantFixed.result) {
				t.Fatalf("fixed result = %v want %v", fixedResult, sc.wantFixed.result)
			}
			if fixedErr != sc.wantFixed.errMsg {
				t.Fatalf("fixed error = %q want %q", fixedErr, sc.wantFixed.errMsg)
			}
			if strings.Contains(sse, "secret upstream boom") {
				t.Fatal("SSE leaked upstream error")
			}
		})
	}

	t.Run("missing server", func(t *testing.T) {
		manager, err := mcp.NewManager(nil)
		if err != nil {
			t.Fatal(err)
		}
		sse := postToolStream(t, manager, `{"server_id":"missing","tool_name":"echo","arguments":{}}`)
		_, _, fixedErr := currentFrontendStreamResult(sse)
		t.Logf("missing server SSE:\n%s", sse)
		t.Logf("fixed parser: success=false error=%q", fixedErr)
		if fixedErr != "Tool execution failed" {
			t.Fatalf("error = %q", fixedErr)
		}
	})
}

func TestExecuteToolStreamUsesChunkTypeAndTerminalResults(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		call       func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error)
		wantEvent  string
		wantData   string
		forbidData string
	}{
		{
			name: "success",
			call: func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error) {
				return &mcpproto.CallToolResult{
					Content: []mcpproto.Content{mcpproto.TextContent{Type: "text", Text: "live-repro-ok"}},
				}, nil
			},
			wantEvent: "event: complete\n",
			wantData:  `"data":"live-repro-ok"`,
		},
		{
			name: "tool error",
			call: func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error) {
				return &mcpproto.CallToolResult{
					IsError: true,
					Content: []mcpproto.Content{mcpproto.TextContent{Type: "text", Text: "tool failed"}},
				}, nil
			},
			wantEvent: "event: error\n",
			wantData:  `"data":"tool failed"`,
		},
		{
			name: "transport failure",
			call: func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error) {
				return nil, errors.New("secret upstream boom")
			},
			wantEvent:  "event: error\n",
			wantData:   `"data":"Tool execution failed"`,
			forbidData: "secret upstream boom",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manager, err := mcp.NewManagerWithInProcessTool("srv", "echo", test.call)
			if err != nil {
				t.Fatal(err)
			}
			body := postToolStream(t, manager, `{"server_id":"srv","tool_name":"echo","arguments":{}}`)
			if strings.Count(body, "event: ") != 1 {
				t.Fatalf("event count = %d\n%s", strings.Count(body, "event: "), body)
			}
			if !strings.Contains(body, test.wantEvent) || !strings.Contains(body, test.wantData) {
				t.Fatalf("body = %s", body)
			}
			if strings.Contains(body, "event: message") {
				t.Fatalf("chunk labeled as message: %s", body)
			}
			if test.forbidData != "" && strings.Contains(body, test.forbidData) {
				t.Fatalf("body leaked remote error: %s", body)
			}
		})
	}
}

func TestExecuteToolStreamEmitsOneErrorWhenTheServerIsMissing(t *testing.T) {
	manager, err := mcp.NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	body := postToolStream(t, manager, `{"server_id":"missing","tool_name":"echo","arguments":{}}`)
	if strings.Count(body, "event: error\n") != 1 || strings.Contains(body, "event: message") {
		t.Fatalf("body = %s", body)
	}
	if !strings.Contains(body, `"type":"error"`) || !strings.Contains(body, "Tool execution failed") {
		t.Fatalf("body = %s", body)
	}
}

func postToolStream(t *testing.T, manager *mcp.Manager, payload string) string {
	t.Helper()
	handler := NewMCPHandler(manager, false)
	req := httptest.NewRequest(http.MethodPost, "/api/mcp/tools/execute/stream", strings.NewReader(payload))
	rec := httptest.NewRecorder()
	handler.ExecuteToolStreamHandler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	return rec.Body.String()
}
