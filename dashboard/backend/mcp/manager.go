package mcp

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

// Manager is the MCP client manager. Server configs are persisted in workflowstore;
// active client connections remain in memory only.
type Manager struct {
	mu              sync.RWMutex
	clients         map[string]*Client
	configs         map[string]*ServerConfig
	ephemeral       map[string]bool
	serverLocks     sync.Map
	store           *workflowstore.Store
	allowStdio      bool
	lifecycleCtx    context.Context
	lifecycleCancel context.CancelFunc
	closed          bool
}

// NewManager loads persisted server configs with the production-safe default:
// local subprocess execution is disabled.
func NewManager(store *workflowstore.Store) (*Manager, error) {
	return NewManagerWithOptions(store, ManagerOptions{})
}

// NewManagerWithOptions loads persisted server configs and installs immutable
// process-level transport policy before any connection can be attempted.
func NewManagerWithOptions(store *workflowstore.Store, options ManagerOptions) (*Manager, error) {
	lifecycleCtx, lifecycleCancel := context.WithCancel(context.Background())
	m := &Manager{
		clients:         make(map[string]*Client),
		configs:         make(map[string]*ServerConfig),
		ephemeral:       make(map[string]bool),
		store:           store,
		allowStdio:      options.AllowStdio,
		lifecycleCtx:    lifecycleCtx,
		lifecycleCancel: lifecycleCancel,
	}
	if store == nil {
		return m, nil
	}
	if err := m.loadConfigs(); err != nil {
		lifecycleCancel()
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
		m.configs[config.ID] = cloneServerConfig(&config)
	}
	return nil
}

func (m *Manager) persistConfig(config *ServerConfig) error {
	if m.store == nil || m.ephemeral[config.ID] {
		return nil
	}
	data, err := json.Marshal(config)
	if err != nil {
		return fmt.Errorf("encode MCP server config: %w", err)
	}
	return m.store.PutMCPServerJSON(config.ID, string(data))
}

// GetServer returns a single server configuration
func (m *Manager) GetServer(id string) (*ServerConfig, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	config, ok := m.configs[id]
	return cloneServerConfig(config), ok
}

// AddServer adds a server configuration and persists it.
func (m *Manager) AddServer(config *ServerConfig) error {
	if err := m.validateRuntimePolicy(config); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.configs[config.ID]; exists {
		return fmt.Errorf("%w: %s", ErrServerExists, config.ID)
	}
	stored := cloneServerConfig(config)
	m.configs[config.ID] = stored
	if err := m.persistConfig(stored); err != nil {
		delete(m.configs, config.ID)
		return err
	}
	return nil
}

// UpsertServer inserts or replaces a server configuration and persists it.
func (m *Manager) UpsertServer(config *ServerConfig) error {
	if err := m.validateRuntimePolicy(config); err != nil {
		return err
	}
	unlock := m.lockServer(config.ID)
	defer unlock()
	m.mu.Lock()
	defer m.mu.Unlock()

	previous, hadPrevious := m.configs[config.ID]
	wasEphemeral := m.ephemeral[config.ID]
	if client, ok := m.clients[config.ID]; ok {
		_ = client.Disconnect()
		delete(m.clients, config.ID)
	}

	delete(m.ephemeral, config.ID)
	stored := cloneServerConfig(config)
	m.configs[config.ID] = stored
	if err := m.persistConfig(stored); err != nil {
		if hadPrevious {
			m.configs[config.ID] = previous
		} else {
			delete(m.configs, config.ID)
		}
		if wasEphemeral {
			m.ephemeral[config.ID] = true
		}
		return err
	}
	return nil
}

// UpsertEphemeralServer installs a process-local server without persisting its
// connection credentials. It is used for the built-in OpenClaw MCP endpoint,
// whose per-process capability must never be written to workflow storage.
func (m *Manager) UpsertEphemeralServer(config *ServerConfig) error {
	if err := m.validateRuntimePolicy(config); err != nil {
		return err
	}
	unlock := m.lockServer(config.ID)
	defer unlock()
	m.mu.Lock()
	defer m.mu.Unlock()

	// Remove any durable row left by an older release before installing the
	// process-local credentials. This guarantees a capability can never survive
	// restart through stale workflow storage.
	if m.store != nil {
		if err := m.store.DeleteMCPServer(config.ID); err != nil && !errors.Is(err, sql.ErrNoRows) {
			return err
		}
	}
	if client, ok := m.clients[config.ID]; ok {
		_ = client.Disconnect()
		delete(m.clients, config.ID)
	}
	m.ephemeral[config.ID] = true
	m.configs[config.ID] = cloneServerConfig(config)
	return nil
}

// UpdateServer updates a server configuration and persists it.
func (m *Manager) UpdateServer(config *ServerConfig) error {
	if err := m.validateRuntimePolicy(config); err != nil {
		return err
	}
	unlock := m.lockServer(config.ID)
	defer unlock()
	m.mu.Lock()
	defer m.mu.Unlock()

	previous, exists := m.configs[config.ID]
	if !exists {
		return fmt.Errorf("%w: %s", ErrServerNotFound, config.ID)
	}

	if client, ok := m.clients[config.ID]; ok {
		_ = client.Disconnect()
		delete(m.clients, config.ID)
	}

	stored := cloneServerConfig(config)
	m.configs[config.ID] = stored
	if err := m.persistConfig(stored); err != nil {
		m.configs[config.ID] = previous
		return err
	}
	return nil
}

// DeleteServer deletes a server configuration and removes it from the store.
func (m *Manager) DeleteServer(id string) error {
	unlock := m.lockServer(id)
	defer unlock()
	m.mu.Lock()
	defer m.mu.Unlock()

	config, exists := m.configs[id]
	if !exists {
		return fmt.Errorf("%w: %s", ErrServerNotFound, id)
	}

	if client, ok := m.clients[id]; ok {
		_ = client.Disconnect()
		delete(m.clients, id)
	}

	delete(m.configs, id)
	wasEphemeral := m.ephemeral[id]
	delete(m.ephemeral, id)
	if m.store == nil || wasEphemeral {
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
	unlock := m.lockServer(id)
	defer unlock()
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return ErrManagerClosed
	}
	config, ok := m.configs[id]
	if !ok {
		m.mu.Unlock()
		return fmt.Errorf("%w: %s", ErrServerNotFound, id)
	}
	if err := m.validateRuntimePolicy(config); err != nil {
		m.mu.Unlock()
		return err
	}

	// Disconnect existing client if any
	if client, ok := m.clients[id]; ok {
		_ = client.Disconnect()
	}

	// Create new client
	client, err := NewClientWithContext(cloneServerConfig(config), m.lifecycleCtx)
	if err != nil {
		m.mu.Unlock()
		return err
	}

	m.clients[id] = client
	m.mu.Unlock()

	// A request timeout controls the handshake while manager shutdown also
	// cancels in-flight setup. Persistent transport resources remain tied to the
	// manager lifecycle rather than the request.
	connectContext, cancel := context.WithCancel(ctx)
	stopLifecycleCancellation := context.AfterFunc(m.lifecycleCtx, cancel)
	defer func() {
		stopLifecycleCancellation()
		cancel()
	}()
	return client.Connect(connectContext)
}

// Disconnect disconnects from the specified server
func (m *Manager) Disconnect(id string) error {
	unlock := m.lockServer(id)
	defer unlock()
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
		return nil, fmt.Errorf("%w: %s", ErrServerNotFound, id)
	}

	client, ok := m.clients[id]
	if !ok {
		return &ServerState{
			Config: cloneServerConfig(config),
			Status: StatusDisconnected,
		}, nil
	}

	state := client.GetState()
	state.Config = cloneServerConfig(state.Config)
	return state, nil
}

// GetAllServerStates returns all server states
func (m *Manager) GetAllServerStates() []*ServerState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	states := make([]*ServerState, 0, len(m.configs))
	for id, config := range m.configs {
		if client, ok := m.clients[id]; ok {
			state := client.GetState()
			state.Config = cloneServerConfig(state.Config)
			states = append(states, state)
		} else {
			states = append(states, &ServerState{
				Config: cloneServerConfig(config),
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
			Error:           "tool execution failed",
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
	if err := m.validateRuntimePolicy(config); err != nil {
		return err
	}
	client, err := NewClientWithContext(cloneServerConfig(config), ctx)
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
			configs = append(configs, cloneServerConfig(config))
		}
	}
	m.mu.RUnlock()

	for _, config := range configs {
		if err := m.validateRuntimePolicy(config); err != nil {
			log.Printf("MCP auto-connect skipped by runtime policy")
			continue
		}
		go func(c *ServerConfig) {
			if err := m.Connect(ctx, c.ID); err != nil {
				log.Printf("MCP auto-connect failed")
			} else {
				log.Printf("MCP auto-connect succeeded")
			}
		}(config)
	}
}

// DisconnectAll disconnects all connections
func (m *Manager) DisconnectAll() {
	m.mu.RLock()
	ids := make([]string, 0, len(m.clients))
	for id := range m.clients {
		ids = append(ids, id)
	}
	m.mu.RUnlock()
	for _, id := range ids {
		_ = m.Disconnect(id)
	}
}

// Close permanently stops the manager, cancels in-flight handshakes and stdio
// subprocesses, and disconnects every client. It is safe to call repeatedly.
func (m *Manager) Close() {
	if m == nil {
		return
	}
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return
	}
	m.closed = true
	m.lifecycleCancel()
	m.mu.Unlock()
	m.DisconnectAll()
}

func (m *Manager) lockServer(id string) func() {
	value, _ := m.serverLocks.LoadOrStore(id, &sync.Mutex{})
	lock := value.(*sync.Mutex)
	lock.Lock()
	return lock.Unlock
}
