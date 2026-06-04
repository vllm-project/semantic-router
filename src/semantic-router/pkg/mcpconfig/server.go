package mcpconfig

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/mark3labs/mcp-go/mcp"
	mcpserver "github.com/mark3labs/mcp-go/server"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// InternalHTTPPath is the loopback MCP endpoint mounted by the router apiserver.
const InternalHTTPPath = "/_internal/mcp/config"

// Server exposes router config MCP tools.
type Server struct {
	mutator *Mutator
	policy  *Policy
	audit   *AuditLog
	httpMCP http.Handler
}

// NewServer builds an MCP config server when cfg.Enabled is true.
func NewServer(configPath string, cfg config.MCPConfigServerConfig) (*Server, error) {
	resolved := cfg.WithDefaults()
	if !resolved.Enabled {
		return nil, fmt.Errorf("mcp config server is disabled")
	}

	paths := resolvePersistencePaths(configPath)
	audit, err := NewAuditLog(defaultAuditLogPath(paths))
	if err != nil {
		return nil, err
	}

	s := &Server{
		mutator: NewMutator(configPath),
		policy:  NewPolicy(resolved),
		audit:   audit,
	}

	mcpServer := mcpserver.NewMCPServer("router-config", "1.0.0")
	s.registerTools(mcpServer)
	s.httpMCP = mcpserver.NewStreamableHTTPServer(mcpServer)
	return s, nil
}

func (s *Server) registerTools(mcpServer *mcpserver.MCPServer) {
	mcpServer.AddTool(getConfigToolDefinition(), s.getConfigTool)
	mcpServer.AddTool(exportConfigJSONToolDefinition(), s.exportConfigJSONTool)
	mcpServer.AddTool(exportConfigYAMLToolDefinition(), s.exportConfigYAMLTool)
	mcpServer.AddTool(validateConfigToolDefinition(), s.validateConfigTool)
	mcpServer.AddTool(diffConfigToolDefinition(), s.diffConfigTool)
	mcpServer.AddTool(applyPatchToolDefinition(), s.applyPatch)
}

func (s *Server) HTTPHandler() http.Handler {
	return s.httpMCP
}

type toolInvocation struct {
	Actor string
	Args  map[string]any
}

func (s *Server) parseInvocation(request mcp.CallToolRequest) (toolInvocation, error) {
	var args map[string]any
	if err := request.BindArguments(&args); err != nil {
		return toolInvocation{}, err
	}
	if args == nil {
		args = map[string]any{}
	}
	actor, _ := args["actor"].(string)
	return toolInvocation{Actor: actor, Args: args}, nil
}

func (s *Server) invoke(
	toolName string,
	invocation toolInvocation,
	destructive bool,
	run func() (any, string, error),
) (*mcp.CallToolResult, error) {
	rawArgs, _ := json.Marshal(invocation.Args)
	argsHash := hashArgs(string(rawArgs))

	if err := s.policy.Check(toolName, invocation.Actor, destructive); err != nil {
		_ = s.audit.Record(AuditEntry{
			Actor:    actorKey(invocation.Actor),
			Tool:     toolName,
			ArgsHash: argsHash,
			Success:  false,
			Error:    err.Error(),
		})
		return mcp.NewToolResultError(err.Error()), nil
	}

	result, version, err := run()
	entry := AuditEntry{
		Actor:         actorKey(invocation.Actor),
		Tool:          toolName,
		ArgsHash:      argsHash,
		Success:       err == nil,
		ConfigVersion: version,
	}
	if err != nil {
		entry.Error = err.Error()
		_ = s.audit.Record(entry)
		return mcp.NewToolResultError(err.Error()), nil
	}
	_ = s.audit.Record(entry)
	return newJSONResult(result)
}

func (s *Server) getConfigTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	_ = ctx
	invocation, err := s.parseInvocation(request)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	return s.invoke(ToolGetConfig, invocation, false, func() (any, string, error) {
		doc, err := s.mutator.GetDocument()
		if err != nil {
			return nil, "", err
		}
		return doc, "", nil
	})
}

func (s *Server) exportConfigJSONTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	_ = ctx
	invocation, err := s.parseInvocation(request)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	return s.invoke(ToolExportConfigJSON, invocation, false, func() (any, string, error) {
		doc, err := s.mutator.GetDocument()
		if err != nil {
			return nil, "", err
		}
		return doc, "", nil
	})
}

func (s *Server) exportConfigYAMLTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	_ = ctx
	invocation, err := s.parseInvocation(request)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	return s.invoke(ToolExportConfigYAML, invocation, false, func() (any, string, error) {
		doc, err := s.mutator.GetDocument()
		if err != nil {
			return nil, "", err
		}
		yamlBytes, err := normalizeRouterConfigDocument(doc)
		if err != nil {
			return nil, "", err
		}
		return map[string]any{"yaml": string(yamlBytes)}, "", nil
	})
}

func (s *Server) validateConfigTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	_ = ctx
	invocation, err := s.parseInvocation(request)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	return s.invoke(ToolValidateConfig, invocation, false, func() (any, string, error) {
		doc, mode, err := documentFromArgs(invocation.Args)
		if err != nil {
			return nil, "", err
		}
		validated, err := s.mutator.ValidateDocument(doc, mode)
		if err != nil {
			return map[string]any{"valid": false, "error": err.Error()}, "", nil
		}
		return map[string]any{"valid": true, "document": validated}, "", nil
	})
}

func (s *Server) diffConfigTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	_ = ctx
	invocation, err := s.parseInvocation(request)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	return s.invoke(ToolDiffConfig, invocation, false, func() (any, string, error) {
		patch, mode, err := patchFromArgs(invocation.Args)
		if err != nil {
			return nil, "", err
		}
		diff, err := s.mutator.DiffDocument(patch, mode)
		if err != nil {
			return nil, "", err
		}
		return diff, "", nil
	})
}

func (s *Server) applyPatch(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	_ = ctx
	invocation, err := s.parseInvocation(request)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	return s.invoke(ToolApplyPatch, invocation, false, func() (any, string, error) {
		patch, mode, err := patchFromArgs(invocation.Args)
		if err != nil {
			return nil, "", err
		}
		dsl, _ := invocation.Args["dsl"].(string)
		result, err := s.mutator.ApplyPatch(patch, mode, dsl)
		if err != nil {
			return nil, "", err
		}
		return result, result.Version, nil
	})
}

func documentFromArgs(args map[string]any) (map[string]any, MutationMode, error) {
	doc, err := objectArg(args, "document")
	if err != nil {
		return nil, "", err
	}
	return doc, mutationModeFromArgs(args), nil
}

func patchFromArgs(args map[string]any) (map[string]any, MutationMode, error) {
	patch, err := objectArg(args, "patch")
	if err != nil {
		return nil, "", err
	}
	return patch, mutationModeFromArgs(args), nil
}

func objectArg(args map[string]any, key string) (map[string]any, error) {
	raw, ok := args[key]
	if !ok {
		return nil, fmt.Errorf("%s is required", key)
	}
	doc, ok := raw.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s must be an object", key)
	}
	return doc, nil
}

func mutationModeFromArgs(args map[string]any) MutationMode {
	merge, ok := args["merge"].(bool)
	if ok && !merge {
		return MutationReplace
	}
	return MutationMerge
}

func newJSONResult(data any) (*mcp.CallToolResult, error) {
	payload, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return nil, err
	}
	return mcp.NewToolResultText(string(payload)), nil
}

// ToolCatalog documents the PR1 MCP tool surface.
func ToolCatalog() []map[string]string {
	tools := CoarseTools()
	out := make([]map[string]string, 0, len(tools))
	for _, tool := range tools {
		out = append(out, map[string]string{
			"name":        tool,
			"description": toolDescription(tool),
		})
	}
	return out
}

func toolDescription(name string) string {
	switch name {
	case ToolGetConfig:
		return "Return the current router config document as JSON."
	case ToolExportConfigJSON:
		return "Export the current router config document as JSON."
	case ToolExportConfigYAML:
		return "Export the current router config document as canonical YAML."
	case ToolValidateConfig:
		return "Validate a router config document."
	case ToolDiffConfig:
		return "Preview a config patch against the current router config."
	case ToolApplyPatch:
		return "Apply a validated config patch to the router config file."
	default:
		return ""
	}
}
