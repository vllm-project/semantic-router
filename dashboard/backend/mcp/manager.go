package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

// Manager is the MCP client manager. Server configs are persisted in workflowstore;
// active client connections remain in memory only.
type Manager struct {
	mu                 sync.RWMutex
	clients            map[string]*Client
	connectAttempts    map[string]*clientConnectAttempt
	configs            map[string]*ServerConfig
	store              *workflowstore.Store
	connectClientFn    func(context.Context, *Client) error
	disconnectClientFn func(*Client) error
}

// NewManager loads persisted server configs from store and returns a new manager.
func NewManager(store *workflowstore.Store) (*Manager, error) {
	m := &Manager{
		clients:         make(map[string]*Client),
		connectAttempts: make(map[string]*clientConnectAttempt),
		configs:         make(map[string]*ServerConfig),
		store:           store,
	}
	if store == nil {
		return m, nil
	}
	if err := m.loadConfigs(); err != nil {
		return nil, fmt.Errorf("load MCP server configs: %w", err)
	}
	return m, nil
}

func (m *Manager) loadConfigs() error {
	if m.store == nil {
		return nil
	}
	rows, err := m.store.ListMCPServerJSON()
	if err != nil {
		return err
	}
	for _, row := range rows {
		var config ServerConfig
		if err := json.Unmarshal([]byte(row), &config); err != nil {
			return fmt.Errorf("decode MCP server config: %w", err)
		}
		if config.ID == "" {
			continue
		}
		m.configs[config.ID] = &config
	}
	return nil
}

func (m *Manager) persistConfig(config *ServerConfig) error {
	if m.store == nil {
		return nil
	}
	data, err := json.Marshal(config)
	if err != nil {
		return fmt.Errorf("encode MCP server config: %w", err)
	}
	return m.store.PutMCPServerJSON(config.ID, string(data))
}

// GetServers returns all server configurations
func (m *Manager) GetServers() []*ServerConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()

	servers := make([]*ServerConfig, 0, len(m.configs))
	for _, config := range m.configs {
		servers = append(servers, config)
	}
	return servers
}

// GetServer returns a single server configuration
func (m *Manager) GetServer(id string) (*ServerConfig, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	config, ok := m.configs[id]
	return config, ok
}

// AddServer adds a server configuration and persists it.
func (m *Manager) AddServer(config *ServerConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.configs[config.ID]; exists {
		return fmt.Errorf("server with ID %s already exists", config.ID)
	}
	if err := m.persistConfig(config); err != nil {
		return err
	}
	m.configs[config.ID] = config
	return nil
}

// UpsertServer inserts or replaces a server configuration and persists it.
func (m *Manager) UpsertServer(config *ServerConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if err := m.persistConfig(config); err != nil {
		return err
	}
	_ = m.disconnectClientLocked(config.ID)

	m.configs[config.ID] = config
	return nil
}

// UpdateServer updates a server configuration and persists it.
func (m *Manager) UpdateServer(config *ServerConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	existing, exists := m.configs[config.ID]
	if !exists {
		return fmt.Errorf("server with ID %s not found", config.ID)
	}
	merged, err := mergeRedactedServerConfig(existing, config)
	if err != nil {
		return fmt.Errorf("update server config: %w", err)
	}

	if err := m.persistConfig(merged); err != nil {
		return err
	}
	_ = m.disconnectClientLocked(config.ID)

	m.configs[config.ID] = merged
	return nil
}

// DeleteServer deletes a server configuration and removes it from the store.
func (m *Manager) DeleteServer(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	_, exists := m.configs[id]
	if !exists {
		return fmt.Errorf("server with ID %s not found", id)
	}

	if m.store != nil {
		if err := m.store.DeleteMCPServer(id); err != nil {
			return err
		}
	}
	_ = m.disconnectClientLocked(id)
	delete(m.configs, id)
	return nil
}

// GetServerStatus returns the server status
func (m *Manager) GetServerStatus(id string) (*ServerState, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	config, ok := m.configs[id]
	if !ok {
		return nil, fmt.Errorf("server with ID %s not found", id)
	}

	client, ok := m.clients[id]
	if !ok {
		return &ServerState{
			Config: config,
			Status: StatusDisconnected,
		}, nil
	}

	return client.GetState(), nil
}

// GetAllServerStates returns all server states
func (m *Manager) GetAllServerStates() []*ServerState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	states := make([]*ServerState, 0, len(m.configs))
	for id, config := range m.configs {
		if client, ok := m.clients[id]; ok {
			states = append(states, client.GetState())
		} else {
			states = append(states, &ServerState{
				Config: config,
				Status: StatusDisconnected,
			})
		}
	}

	return states
}

// GetAllTools returns all tools from connected servers
func (m *Manager) GetAllTools() []Tool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var tools []Tool
	for id, client := range m.clients {
		if client.GetStatus() != StatusConnected {
			continue
		}

		config := m.configs[id]
		for _, tool := range client.GetTools() {
			tools = append(tools, Tool{
				ToolDefinition: tool,
				ServerID:       id,
				ServerName:     config.Name,
			})
		}
	}

	return tools
}

// ExecuteTool executes a tool
func (m *Manager) ExecuteTool(ctx context.Context, serverID, toolName string, arguments json.RawMessage) (*ToolResult, error) {
	m.mu.RLock()
	client, ok := m.clients[serverID]
	m.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("server %s not connected", serverID)
	}

	if client.GetStatus() != StatusConnected {
		return nil, fmt.Errorf("server %s not connected", serverID)
	}

	start := time.Now()
	result, err := client.CallTool(ctx, toolName, arguments)
	elapsed := time.Since(start)

	if err != nil {
		return nil, fmt.Errorf("execute MCP tool: %w", err)
	}

	// Convert content
	var content interface{}
	if len(result.Content) > 0 {
		if len(result.Content) == 1 && result.Content[0].Type == "text" {
			content = result.Content[0].Text
		} else {
			content = result.Content
		}
	}

	return &ToolResult{
		Success:         !result.IsError,
		Result:          content,
		ExecutionTimeMs: elapsed.Milliseconds(),
	}, nil
}

// ExecuteToolStreaming executes a tool with streaming
func (m *Manager) ExecuteToolStreaming(ctx context.Context, serverID, toolName string, arguments json.RawMessage, onChunk func(StreamChunk) error) error {
	m.mu.RLock()
	client, ok := m.clients[serverID]
	m.mu.RUnlock()

	if !ok {
		return fmt.Errorf("server %s not connected", serverID)
	}

	if client.GetStatus() != StatusConnected {
		return fmt.Errorf("server %s not connected", serverID)
	}

	return client.CallToolStreaming(ctx, toolName, arguments, onChunk)
}

func (m *Manager) resolveConnectionTestConfig(config *ServerConfig) (*ServerConfig, error) {
	if config == nil {
		return nil, fmt.Errorf("MCP server config is required")
	}

	m.mu.RLock()
	existing, exists := m.configs[config.ID]
	if !exists {
		existing = &ServerConfig{}
	}
	resolved, err := mergeRedactedServerConfig(existing, config)
	m.mu.RUnlock()
	if err != nil {
		return nil, fmt.Errorf("resolve connection test config: %w", err)
	}
	return resolved, nil
}

// TestConnection tests the connection without persisting the submitted config.
// Redacted placeholders from the edit dialog are resolved against the stored
// server before the temporary client is created.
func (m *Manager) TestConnection(ctx context.Context, config *ServerConfig) error {
	resolved, err := m.resolveConnectionTestConfig(config)
	if err != nil {
		return err
	}
	client, err := NewClient(resolved)
	if err != nil {
		return err
	}
	defer func() { _ = client.Disconnect() }()

	return client.Connect(ctx)
}

// ConnectEnabled connects to all enabled servers
func (m *Manager) ConnectEnabled(ctx context.Context) {
	m.mu.RLock()
	configs := make([]*ServerConfig, 0)
	for _, config := range m.configs {
		if config.Enabled {
			configs = append(configs, config)
		}
	}
	m.mu.RUnlock()

	for _, config := range configs {
		go func(c *ServerConfig) {
			if err := m.Connect(ctx, c.ID); err != nil {
				log.Printf("[MCP-Manager] ConnectEnabled failed: error_class=%T", err)
			} else {
				log.Printf("[MCP-Manager] ConnectEnabled succeeded")
			}
		}(config)
	}
}

// DisconnectAll disconnects all connections
func (m *Manager) DisconnectAll() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for id := range m.clients {
		_ = m.disconnectClientLocked(id)
	}
}
