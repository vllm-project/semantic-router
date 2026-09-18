package kvtransfer

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

const defaultAddressRegistryTTL = time.Hour

// AddressRecord is the Redis/Valkey payload for kv_addr:{namespace}:{session_id}.
type AddressRecord struct {
	SessionID string    `json:"session_id"`
	SourcePod string    `json:"source_pod"`
	Model     string    `json:"model"`
	Namespace string    `json:"namespace"`
	TurnCount int       `json:"turn_count"`
	UpdatedAt time.Time `json:"updated_at"`
}

// AddressRegistry stores the last known KV-cache location for a routed session.
type AddressRegistry interface {
	Write(ctx context.Context, record AddressRecord) error
	Lookup(ctx context.Context, namespace, sessionID string) (*AddressRecord, error)
	Close() error
}

// NoopAddressRegistry disables cross-model KV address tracking.
type NoopAddressRegistry struct{}

func (NoopAddressRegistry) Write(context.Context, AddressRecord) error { return nil }
func (NoopAddressRegistry) Lookup(context.Context, string, string) (*AddressRecord, error) {
	return nil, nil
}
func (NoopAddressRegistry) Close() error { return nil }

// MemoryAddressRegistry is an in-process registry for unit tests.
type MemoryAddressRegistry struct {
	mu      sync.RWMutex
	entries map[string]AddressRecord
}

func NewMemoryAddressRegistry() *MemoryAddressRegistry {
	return &MemoryAddressRegistry{entries: make(map[string]AddressRecord)}
}

func (r *MemoryAddressRegistry) Write(_ context.Context, record AddressRecord) error {
	if r == nil {
		return nil
	}
	key := AddressKey(record.Namespace, record.SessionID)
	r.mu.Lock()
	r.entries[key] = record
	r.mu.Unlock()
	return nil
}

func (r *MemoryAddressRegistry) Lookup(_ context.Context, namespace, sessionID string) (*AddressRecord, error) {
	if r == nil {
		return nil, nil
	}
	key := AddressKey(namespace, sessionID)
	r.mu.RLock()
	record, ok := r.entries[key]
	r.mu.RUnlock()
	if !ok {
		return nil, nil
	}
	copy := record
	return &copy, nil
}

func (r *MemoryAddressRegistry) Close() error { return nil }

// AddressKey returns the canonical kv_addr registry key for a tenant/session pair.
func AddressKey(namespace, sessionID string) string {
	return fmt.Sprintf(
		"kv_addr:%s:%s",
		escapeAddressKeyPart(namespace),
		escapeAddressKeyPart(sessionID),
	)
}

func escapeAddressKeyPart(part string) string {
	return strings.ReplaceAll(part, ":", "%3A")
}

func validateAddressRecord(record AddressRecord) error {
	if strings.TrimSpace(record.SessionID) == "" {
		return fmt.Errorf("session_id is required")
	}
	if strings.TrimSpace(record.SourcePod) == "" {
		return fmt.Errorf("source_pod is required")
	}
	if strings.TrimSpace(record.Model) == "" {
		return fmt.Errorf("model is required")
	}
	if record.TurnCount < 0 {
		return fmt.Errorf("turn_count cannot be negative")
	}
	return nil
}

func encodeAddressRecord(record AddressRecord) ([]byte, error) {
	payload, err := json.Marshal(record)
	if err != nil {
		return nil, fmt.Errorf("encode kv address record: %w", err)
	}
	return payload, nil
}

func decodeAddressRecord(payload []byte) (AddressRecord, error) {
	var record AddressRecord
	if err := json.Unmarshal(payload, &record); err != nil {
		return AddressRecord{}, fmt.Errorf("decode kv address record: %w", err)
	}
	return record, nil
}
