package mcpconfig

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// AuditEntry records one MCP config tool invocation.
type AuditEntry struct {
	Timestamp     string `json:"timestamp"`
	Actor         string `json:"actor"`
	Tool          string `json:"tool"`
	ArgsHash      string `json:"args_hash"`
	Success       bool   `json:"success"`
	Error         string `json:"error,omitempty"`
	ConfigVersion string `json:"config_version,omitempty"`
}

// AuditLog appends durable audit records for MCP config tool calls.
type AuditLog struct {
	path string
	mu   sync.Mutex
}

func NewAuditLog(path string) (*AuditLog, error) {
	if path == "" {
		return &AuditLog{}, nil
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create audit log directory: %w", err)
	}
	return &AuditLog{path: path}, nil
}

func (a *AuditLog) Record(entry AuditEntry) error {
	if a == nil || a.path == "" {
		return nil
	}
	if entry.Timestamp == "" {
		entry.Timestamp = time.Now().UTC().Format(time.RFC3339Nano)
	}

	payload, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	f, err := os.OpenFile(a.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := f.Write(append(payload, '\n')); err != nil {
		return err
	}
	return nil
}

func hashArgs(raw string) string {
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:8])
}
