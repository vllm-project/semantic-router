package mcpserver

import (
	"context"
	"testing"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestConfigMCPServer_ToolsExistAndWork(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.KeywordRules = []config.KeywordRule{{Name: "k1", Keywords: []string{"x"}}}
	cfg.Decisions = []config.Decision{{Name: "d1"}}
	cfg.ModelConfig = map[string]config.ModelParams{"m1": {}}

	srv := New(&StaticConfigProvider{Cfg: cfg})
	c, err := client.NewInProcessClient(srv.mcpServer)
	if err != nil {
		t.Fatalf("NewInProcessClient: %v", err)
	}
	if err := c.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}

	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{Name: "test", Version: "0"}
	_, err = c.Initialize(context.Background(), initReq)
	if err != nil {
		t.Fatalf("Initialize: %v", err)
	}

	toolsResp, err := c.ListTools(context.Background(), mcp.ListToolsRequest{})
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	if len(toolsResp.Tools) == 0 {
		t.Fatalf("expected tools")
	}

	call := mcp.CallToolRequest{}
	call.Params.Name = "list_decision_names"
	res, err := c.CallTool(context.Background(), call)
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if res == nil || len(res.Content) == 0 {
		t.Fatalf("expected content")
	}
}
