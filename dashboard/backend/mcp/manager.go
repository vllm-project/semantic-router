package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

// Manager is the MCP client manager. Server configs are persisted in workflowstore;
// active client connections remain in memory only.
type Manager struct {
	mu      sync.RWMutex
	clients map[string]*Client
	configs map[string]*ServerConfig
	store   *workflowstore.Store
}

// NewManager loads persisted server configs from store and returns a new manager.
func NewManager(store *workflowstore.Store) (*Manager, error) {
	m := &Manager{
		clients: make(map[string]*Client),
		configs: make(map[string]*ServerConfig),
		store:   store,
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
	m.configs[config.ID] = config
	if err := m.persistConfig(config); err != nil {
		delete(m.configs, config.ID)
		return err
	}
	return nil
}

// UpsertServer inserts or replaces a server configuration and persists it.
func (m *Manager) UpsertServer(config *ServerConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if client, ok := m.clients[config.ID]; ok {
		_ = client.Disconnect()
		delete(m.clients, config.ID)
	}

	m.configs[config.ID] = config
	return m.persistConfig(config)
}

// UpdateServer updates a server configuration and persists it.
func (m *Manager) UpdateServer(config *ServerConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.configs[config.ID]; !exists {
		return fmt.Errorf("server with ID %s not found", config.ID)
	}

	if client, ok := m.clients[config.ID]; ok {
		_ = client.Disconnect()
		delete(m.clients, config.ID)
	}

	m.configs[config.ID] = config
	return m.persistConfig(config)
}

// DeleteServer deletes a server configuration and removes it from the store.
func (m *Manager) DeleteServer(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	config, exists := m.configs[id]
	if !exists {
		return fmt.Errorf("server with ID %s not found", id)
	}

	if client, ok := m.clients[id]; ok {
		_ = client.Disconnect()
		delete(m.clients, id)
	}

	delete(m.configs, id)
	if m.store == nil {
		return nil
	}
	if err := m.store.DeleteMCPServer(id); err != nil {
		m.configs[id] = config
		return err
	}
	return nil
}

// Connect establishes connection to the specified server
func (m *Manager) Connect(ctx context.Context, id string) error {
	m.mu.Lock()
	config, ok := m.configs[id]
	if !ok {
		m.mu.Unlock()
		return fmt.Errorf("server with ID %s not found", id)
	}

	// Disconnect existing client if any
	if client, ok := m.clients[id]; ok {
		_ = client.Disconnect()
	}

	// Create new client
	client, err := NewClient(config)
	if err != nil {
		m.mu.Unlock()
		return err
	}

	m.clients[id] = client
	m.mu.Unlock()

	// Connect
	return client.Connect(ctx)
}

// Disconnect disconnects from the specified server
func (m *Manager) Disconnect(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	client, ok := m.clients[id]
	if !ok {
		return nil
	}

	err := client.Disconnect()
	delete(m.clients, id)

	return err
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
		return &ToolResult{
			Success:         false,
			Error:           err.Error(),
			ExecutionTimeMs: elapsed.Milliseconds(),
		}, nil
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

// TestConnection tests the connection
func (m *Manager) TestConnection(ctx context.Context, config *ServerConfig) error {
	client, err := NewClient(config)
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
				fmt.Printf("Failed to connect to MCP server %s: %v\n", c.Name, err)
			} else {
				fmt.Printf("Connected to MCP server %s\n", c.Name)
			}
		}(config)
	}
}

// DisconnectAll disconnects all connections
func (m *Manager) DisconnectAll() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for id, client := range m.clients {
		_ = client.Disconnect()
		delete(m.clients, id)
	}
}
