package mcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"
	"time"

	mcpsdk "github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const transportHelperEnvironment = "VLLM_SR_MCP_TRANSPORT_HELPER"

func classificationToolResult(request mcpsdk.CallToolRequest) *mcpsdk.CallToolResult {
	if request.GetString("text", "") != "classify this" || !request.GetBool("with_probabilities", false) {
		return mcpsdk.NewToolResultError("classification arguments missing")
	}
	payload, _ := json.Marshal(map[string]interface{}{
		"class": 1, "confidence": 0.8, "probabilities": []float64{0.1, 0.8, 0.1},
		"model": "fixture-model", "use_reasoning": request.GetBool("reasoning", false),
	})
	return mcpsdk.NewToolResultText(string(payload))
}

func classificationTool() mcpsdk.Tool {
	return mcpsdk.NewTool("classify_text",
		mcpsdk.WithString("text"), mcpsdk.WithBoolean("with_probabilities"),
		mcpsdk.WithBoolean("reasoning"))
}

func httpClassificationClient(t *testing.T) MCPClient {
	t.Helper()
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		var result interface{}
		switch request.URL.Path {
		case "/initialize":
			result = map[string]interface{}{}
		case "/tools/list":
			result = mcpsdk.ListToolsResult{Tools: []mcpsdk.Tool{classificationTool()}}
		case "/resources/list":
			result = mcpsdk.ListResourcesResult{}
		case "/prompts/list":
			result = mcpsdk.ListPromptsResult{}
		case "/tools/call":
			var call mcpsdk.CallToolRequest
			if err := json.NewDecoder(request.Body).Decode(&call); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			result = classificationToolResult(call)
		default:
			http.NotFound(w, request)
			return
		}
		_ = json.NewEncoder(w).Encode(result)
	}))
	t.Cleanup(endpoint.Close)
	client := NewHTTPClient("classification", ClientConfig{URL: endpoint.URL, Timeout: 5 * time.Second})
	if err := client.Connect(); err != nil {
		t.Fatalf("connect HTTP transport: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

func stdioClassificationClient(t *testing.T) MCPClient {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	client := NewStdioClient("classification", ClientConfig{
		Command: executable, Args: []string{"-test.run=^TestMCPToolTransports$"},
		Env: map[string]string{transportHelperEnvironment: "1"}, Timeout: 5 * time.Second,
	})
	// Exercise the real child-process transport with synchronous capability
	// discovery; background discovery/readiness is a separate lifecycle contract.
	if err := client.createMCPClient(); err != nil {
		t.Fatalf("create stdio transport: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := client.mcpClient.Initialize(ctx, mcpsdk.InitializeRequest{}); err != nil {
		t.Fatalf("initialize stdio transport: %v", err)
	}
	if err := client.loadCapabilities(ctx); err != nil {
		t.Fatalf("discover stdio tools: %v", err)
	}
	return client
}

func TestMCPToolTransports(t *testing.T) {
	if os.Getenv(transportHelperEnvironment) == "1" {
		fixture := server.NewMCPServer("classification-fixture", "1",
			server.WithToolCapabilities(true),
			server.WithResourceCapabilities(false, false),
			server.WithPromptCapabilities(false))
		fixture.AddTool(classificationTool(), func(_ context.Context, call mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			return classificationToolResult(call), nil
		})
		if err := server.ServeStdio(fixture); err != nil {
			t.Fatal(err)
		}
		return
	}
	for name, connect := range map[string]func(*testing.T) MCPClient{
		"http": httpClassificationClient, "stdio": stdioClassificationClient,
	} {
		t.Run(name, func(t *testing.T) {
			client := connect(t)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			for _, reasoning := range []bool{false, true} {
				result, err := client.CallTool(ctx, "classify_text", map[string]interface{}{
					"text": "classify this", "with_probabilities": true, "reasoning": reasoning,
				})
				if err != nil || result == nil || result.IsError || len(result.Content) != 1 {
					t.Fatalf("classification result = %#v, error = %v", result, err)
				}
				content, ok := result.Content[0].(mcpsdk.TextContent)
				if !ok {
					t.Fatalf("classification content = %T", result.Content[0])
				}
				var actual map[string]interface{}
				if err := json.Unmarshal([]byte(content.Text), &actual); err != nil {
					t.Fatal(err)
				}
				expected := map[string]interface{}{
					"class": float64(1), "confidence": 0.8,
					"probabilities": []interface{}{0.1, 0.8, 0.1},
					"model":         "fixture-model", "use_reasoning": reasoning,
				}
				if !reflect.DeepEqual(actual, expected) {
					t.Fatalf("transport changed classification payload: got %#v, want %#v", actual, expected)
				}
			}
			result, err := client.CallTool(ctx, "classify_text", map[string]interface{}{"text": "invalid"})
			if err != nil || result == nil || !result.IsError {
				t.Fatalf("tool error was lost: result=%#v, error=%v", result, err)
			}
			if _, err := client.CallTool(ctx, "unlisted-tool", nil); err == nil {
				t.Fatal("transport accepted an undiscovered tool")
			}
		})
	}
}
