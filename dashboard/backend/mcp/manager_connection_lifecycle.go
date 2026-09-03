package mcp

import (
	"context"
	"errors"
	"fmt"
)

var errConnectionSuperseded = errors.New("server configuration changed while connecting")

type clientConnectAttempt struct {
	client *Client
	cancel context.CancelFunc
}

// Connect establishes a connection to the specified server generation.
func (m *Manager) Connect(ctx context.Context, id string) error {
	m.mu.Lock()
	config, ok := m.configs[id]
	if !ok {
		m.mu.Unlock()
		return fmt.Errorf("server with ID %s not found", id)
	}

	// Cancel and close any previous generation before publishing its replacement.
	_ = m.disconnectClientLocked(id)

	client, err := NewClient(config)
	if err != nil {
		m.mu.Unlock()
		return err
	}

	connectCtx, cancel := context.WithCancel(ctx)
	attempt := &clientConnectAttempt{client: client, cancel: cancel}
	m.clients[id] = client
	m.connectAttempts[id] = attempt
	m.mu.Unlock()

	connectErr := m.connectClient(connectCtx, client)
	cancel()

	m.mu.Lock()
	current := m.clients[id] == client && m.connectAttempts[id] == attempt
	if current {
		delete(m.connectAttempts, id)
		m.mu.Unlock()
		return connectErr
	}
	m.mu.Unlock()

	// A concurrent update, delete, disconnect, or replacement won the race. The
	// SDK may not have observed cancellation before publishing its transport, so
	// close this stale generation again after Connect returns.
	_ = m.disconnectClient(client)
	return errConnectionSuperseded
}

// Disconnect disconnects from the specified server.
func (m *Manager) Disconnect(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.disconnectClientLocked(id)
}

func (m *Manager) connectClient(ctx context.Context, client *Client) error {
	if m.connectClientFn != nil {
		return m.connectClientFn(ctx, client)
	}
	return client.Connect(ctx)
}

func (m *Manager) disconnectClient(client *Client) error {
	if m.disconnectClientFn != nil {
		return m.disconnectClientFn(client)
	}
	return client.Disconnect()
}

// disconnectClientLocked cancels an in-flight Connect before closing and
// removing the published client generation. The manager mutex must be held.
func (m *Manager) disconnectClientLocked(id string) error {
	if attempt, ok := m.connectAttempts[id]; ok {
		attempt.cancel()
		delete(m.connectAttempts, id)
	}
	client, ok := m.clients[id]
	if !ok {
		return nil
	}
	delete(m.clients, id)
	return m.disconnectClient(client)
}
