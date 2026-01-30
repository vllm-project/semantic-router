package mcpserver

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"sync"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// DefaultAddr is the default listen address for the internal MCP config server.
const DefaultAddr = ":8088"

// DefaultPath is the default HTTP base path (best-effort).
const DefaultPath = "/mcp"

type ConfigProvider interface {
	Get() *config.RouterConfig
}

// GlobalConfigProvider returns the process-global router config.
// It is safe to use when the router config is managed via config.Replace(...).
type GlobalConfigProvider struct{}

func (p *GlobalConfigProvider) Get() *config.RouterConfig { return config.Get() }

type StaticConfigProvider struct {
	Cfg *config.RouterConfig
}

func (p *StaticConfigProvider) Get() *config.RouterConfig { return p.Cfg }

// Server wraps an MCP server instance that exposes config-oriented tools.
type Server struct {
	cfgProvider ConfigProvider
	persistPath string

	mu         sync.Mutex
	mcpServer  *server.MCPServer
	httpServer *server.StreamableHTTPServer
	once       sync.Once
}

func New(cfgProvider ConfigProvider) *Server {
	return NewWithPersist(cfgProvider, "")
}

// NewWithPersist creates an MCP server. If persistPath is non-empty, update/add/delete
// tools will write the updated full config back to disk.
func NewWithPersist(cfgProvider ConfigProvider, persistPath string) *Server {
	m := server.NewMCPServer(
		"semantic-router-config",
		"0.2.0",
		server.WithToolCapabilities(true),
	)

	s := &Server{cfgProvider: cfgProvider, persistPath: persistPath, mcpServer: m}
	s.registerTools()
	return s
}

func (s *Server) registerTools() {
	// v0.1 tools (kept for compatibility)
	s.mcpServer.AddTool(mcp.Tool{
		Name:        "get_current_config_yaml",
		Description: "Return the current Semantic Router config as YAML (read-only).",
		InputSchema: mcp.ToolInputSchema{Type: "object", Properties: map[string]any{}},
	}, s.handleGetCurrentConfigYAML)

	s.mcpServer.AddTool(mcp.Tool{
		Name:        "validate_config_yaml",
		Description: "Validate a proposed Semantic Router config YAML. Returns validation errors if any.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"yaml": map[string]any{
					"type":        "string",
					"description": "The full router config in YAML form.",
				},
			},
			Required: []string{"yaml"},
		},
	}, s.handleValidateConfigYAML)

	s.mcpServer.AddTool(mcp.Tool{
		Name:        "list_signal_names",
		Description: "List configured signal names (keywords, embeddings, fact-check, preference, etc.) from the current config.",
		InputSchema: mcp.ToolInputSchema{Type: "object", Properties: map[string]any{}},
	}, s.handleListSignalNames)

	s.mcpServer.AddTool(mcp.Tool{
		Name:        "list_decision_names",
		Description: "List configured decision names from the current config.",
		InputSchema: mcp.ToolInputSchema{Type: "object", Properties: map[string]any{}},
	}, s.handleListDecisionNames)

	s.mcpServer.AddTool(mcp.Tool{
		Name:        "list_model_names",
		Description: "List configured provider model names from the current config.",
		InputSchema: mcp.ToolInputSchema{Type: "object", Properties: map[string]any{}},
	}, s.handleListModelNames)

	// v0.2 CRUD tools
	s.mcpServer.AddTool(mcp.Tool{
		Name:        "list_kinds",
		Description: "List supported entity kinds for config CRUD (decision, keyword_rule, model, etc.)",
		InputSchema: mcp.ToolInputSchema{Type: "object", Properties: map[string]any{}},
	}, s.handleListKinds)

	s.mcpServer.AddTool(mcp.Tool{
		Name:        "list",
		Description: "List entity names for a given kind.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"kind": map[string]any{"type": "string"},
			},
			Required: []string{"kind"},
		},
	}, s.handleList)

	s.mcpServer.AddTool(mcp.Tool{
		Name:        "get",
		Description: "Get an entity by kind+name. Returns YAML.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"kind": map[string]any{"type": "string"},
				"name": map[string]any{"type": "string"},
			},
			Required: []string{"kind", "name"},
		},
	}, s.handleGet)

	s.mcpServer.AddTool(mcp.Tool{
		Name:        "add",
		Description: "Add an entity. item_yaml is the YAML representation of the entity. For kind=model, item_yaml must be {name: <modelName>, params: <ModelParams>}.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"kind":      map[string]any{"type": "string"},
				"item_yaml": map[string]any{"type": "string"},
			},
			Required: []string{"kind", "item_yaml"},
		},
	}, s.handleAdd)

	s.mcpServer.AddTool(mcp.Tool{
		Name:        "update",
		Description: "Update an entity by kind+name. item_yaml is the YAML representation of the entity (name will be forced to match). For kind=model, item_yaml is ModelParams YAML.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"kind":      map[string]any{"type": "string"},
				"name":      map[string]any{"type": "string"},
				"item_yaml": map[string]any{"type": "string"},
			},
			Required: []string{"kind", "name", "item_yaml"},
		},
	}, s.handleUpdate)

	s.mcpServer.AddTool(mcp.Tool{
		Name:        "delete",
		Description: "Delete an entity by kind+name.",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"kind": map[string]any{"type": "string"},
				"name": map[string]any{"type": "string"},
			},
			Required: []string{"kind", "name"},
		},
	}, s.handleDelete)
}

func (s *Server) handleListKinds(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	b, _ := yaml.Marshal(supportedKinds())
	return mcp.NewToolResultText(string(b)), nil
}

func (s *Server) handleList(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	kind, err := req.RequireString("kind")
	if err != nil {
		return nil, err
	}
	cfg := s.cfgProvider.Get()
	if cfg == nil {
		return mcp.NewToolResultText("[]"), nil
	}
	names, err := listNames(kind, cfg)
	if err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: %v", err)), nil
	}
	b, _ := yaml.Marshal(names)
	return mcp.NewToolResultText(string(b)), nil
}

func (s *Server) handleGet(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	kind, err := req.RequireString("kind")
	if err != nil {
		return nil, err
	}
	name, err := req.RequireString("name")
	if err != nil {
		return nil, err
	}
	cfg := s.cfgProvider.Get()
	if cfg == nil {
		return mcp.NewToolResultText("error: config is nil"), nil
	}
	entity, err := getEntityYAML(kind, name, cfg)
	if err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: %v", err)), nil
	}
	b, _ := yaml.Marshal(entity)
	return mcp.NewToolResultText(string(b)), nil
}

func (s *Server) handleAdd(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	kind, err := req.RequireString("kind")
	if err != nil {
		return nil, err
	}
	itemYAML, err := req.RequireString("item_yaml")
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	cur := s.cfgProvider.Get()
	if cur == nil {
		return mcp.NewToolResultText("error: config is nil"), nil
	}
	newCfg, err := cloneConfig(cur)
	if err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: %v", err)), nil
	}

	name, err := addEntity(kind, newCfg, itemYAML)
	if err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: %v", err)), nil
	}

	config.Replace(newCfg)
	if err := persistConfig(s.persistPath, newCfg); err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: persist failed: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("ok: added %s %q", normalizeKind(kind), name)), nil
}

func (s *Server) handleUpdate(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	kind, err := req.RequireString("kind")
	if err != nil {
		return nil, err
	}
	name, err := req.RequireString("name")
	if err != nil {
		return nil, err
	}
	itemYAML, err := req.RequireString("item_yaml")
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	cur := s.cfgProvider.Get()
	if cur == nil {
		return mcp.NewToolResultText("error: config is nil"), nil
	}
	newCfg, err := cloneConfig(cur)
	if err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: %v", err)), nil
	}

	if err := updateEntity(kind, name, newCfg, itemYAML); err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: %v", err)), nil
	}

	config.Replace(newCfg)
	if err := persistConfig(s.persistPath, newCfg); err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: persist failed: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("ok: updated %s %q", normalizeKind(kind), name)), nil
}

func (s *Server) handleDelete(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	kind, err := req.RequireString("kind")
	if err != nil {
		return nil, err
	}
	name, err := req.RequireString("name")
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	cur := s.cfgProvider.Get()
	if cur == nil {
		return mcp.NewToolResultText("error: config is nil"), nil
	}
	newCfg, err := cloneConfig(cur)
	if err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: %v", err)), nil
	}

	if err := deleteEntity(kind, name, newCfg); err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: %v", err)), nil
	}

	config.Replace(newCfg)
	if err := persistConfig(s.persistPath, newCfg); err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("error: persist failed: %v", err)), nil
	}

	return mcp.NewToolResultText(fmt.Sprintf("ok: deleted %s %q", normalizeKind(kind), name)), nil
}

func (s *Server) handleGetCurrentConfigYAML(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	cfg := s.cfgProvider.Get()
	if cfg == nil {
		return mcp.NewToolResultText("config is nil"), nil
	}

	b, err := yaml.Marshal(cfg)
	if err != nil {
		return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{mcp.TextContent{Type: "text", Text: err.Error()}}}, nil
	}

	return mcp.NewToolResultText(string(b)), nil
}

func (s *Server) handleValidateConfigYAML(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	yamlStr, err := req.RequireString("yaml")
	if err != nil {
		return nil, err
	}

	var cfg config.RouterConfig
	dec := yaml.NewDecoder(bytes.NewBufferString(yamlStr))
	dec.KnownFields(true)
	if err := dec.Decode(&cfg); err != nil {
		return mcp.NewToolResultText(fmt.Sprintf("YAML parse error: %v", err)), nil
	}

	// Reuse existing validation logic by round-tripping through Parse is hard without a file.
	// For now: rely on YAML decode + minimal semantic checks.
	if cfg.DefaultModel == "" && cfg.ModelConfig == nil && len(cfg.VLLMEndpoints) == 0 {
		return mcp.NewToolResultText("validation warning: config may be incomplete (no default_model/model_config/vllm_endpoints detected)"), nil
	}

	return mcp.NewToolResultText("ok"), nil
}

func (s *Server) handleListSignalNames(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	cfg := s.cfgProvider.Get()
	if cfg == nil {
		return mcp.NewToolResultText("[]"), nil
	}

	// Signals are spread across config fields. We do best-effort extraction.
	names := make([]string, 0, 64)
	for _, r := range cfg.KeywordRules {
		if r.Name != "" {
			names = append(names, r.Name)
		}
	}
	for _, r := range cfg.EmbeddingRules {
		if r.Name != "" {
			names = append(names, r.Name)
		}
	}
	for _, r := range cfg.FactCheckRules {
		if r.Name != "" {
			names = append(names, r.Name)
		}
	}
	for _, r := range cfg.UserFeedbackRules {
		if r.Name != "" {
			names = append(names, r.Name)
		}
	}
	for _, r := range cfg.PreferenceRules {
		if r.Name != "" {
			names = append(names, r.Name)
		}
	}
	for _, r := range cfg.LanguageRules {
		if r.Name != "" {
			names = append(names, r.Name)
		}
	}
	for _, r := range cfg.ContextRules {
		if r.Name != "" {
			names = append(names, r.Name)
		}
	}
	for _, r := range cfg.LatencyRules {
		if r.Name != "" {
			names = append(names, r.Name)
		}
	}
	for _, r := range cfg.ComplexityRules {
		if r.Name != "" {
			names = append(names, r.Name)
		}
	}

	b, _ := yaml.Marshal(names)
	return mcp.NewToolResultText(string(b)), nil
}

func (s *Server) handleListDecisionNames(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	cfg := s.cfgProvider.Get()
	if cfg == nil {
		return mcp.NewToolResultText("[]"), nil
	}

	names := make([]string, 0, len(cfg.Decisions))
	for _, d := range cfg.Decisions {
		if d.Name != "" {
			names = append(names, d.Name)
		}
	}
	b, _ := yaml.Marshal(names)
	return mcp.NewToolResultText(string(b)), nil
}

func (s *Server) handleListModelNames(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	cfg := s.cfgProvider.Get()
	if cfg == nil {
		return mcp.NewToolResultText("[]"), nil
	}

	names := make([]string, 0, len(cfg.ModelConfig))
	for name := range cfg.ModelConfig {
		names = append(names, name)
	}
	b, _ := yaml.Marshal(names)
	return mcp.NewToolResultText(string(b)), nil
}

func (s *Server) StartHTTP(addr string, path string) error {
	if addr == "" {
		addr = DefaultAddr
	}
	if path == "" {
		path = DefaultPath
	}

	var err error
	s.once.Do(func() {
		s.httpServer = server.NewStreamableHTTPServer(s.mcpServer)
		_ = path // base path currently not configurable in mcp-go StreamableHTTPServer
		err = s.httpServer.Start(addr)
	})
	return err
}

func (s *Server) Handler() http.Handler {
	if s.httpServer == nil {
		s.httpServer = server.NewStreamableHTTPServer(s.mcpServer)
	}
	return s.httpServer
}
