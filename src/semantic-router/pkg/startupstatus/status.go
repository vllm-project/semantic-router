package startupstatus

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// State captures router startup readiness beyond process-level health.
type State struct {
	Phase            string   `json:"phase"`
	Ready            bool     `json:"ready"`
	Message          string   `json:"message,omitempty"`
	DownloadingModel string   `json:"downloading_model,omitempty"`
	PendingModels    []string `json:"pending_models,omitempty"`
	ReadyModels      int      `json:"ready_models,omitempty"`
	TotalModels      int      `json:"total_models,omitempty"`
	UpdatedAt        string   `json:"updated_at,omitempty"`
}

// Writer persists startup state to a JSON file that can be read by the dashboard.
type Writer struct {
	path string
	mu   sync.Mutex
}

// NewWriter creates a writer using a router config path.
func NewWriter(configPath string) *Writer {
	return &Writer{path: StatusPathFromConfigPath(configPath)}
}

// StatusPathFromConfigPath returns the runtime status file path next to router-config.yaml.
func StatusPathFromConfigPath(configPath string) string {
	return filepath.Join(filepath.Dir(configPath), "router-runtime.json")
}

// Write persists the provided state atomically.
func (w *Writer) Write(state State) error {
	if w == nil || w.path == "" {
		return nil
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	state.UpdatedAt = time.Now().UTC().Format(time.RFC3339)

	if err := os.MkdirAll(filepath.Dir(w.path), 0o755); err != nil {
		return fmt.Errorf("create startup status dir: %w", err)
	}

	payload, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal startup status: %w", err)
	}

	tmpPath := w.path + ".tmp"
	if err := os.WriteFile(tmpPath, payload, 0o644); err != nil {
		return fmt.Errorf("write startup status temp file: %w", err)
	}

	if err := os.Rename(tmpPath, w.path); err != nil {
		return fmt.Errorf("replace startup status file: %w", err)
	}

	return nil
}

// Load reads a runtime status file from disk.
func Load(path string) (*State, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var state State
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("decode startup status: %w", err)
	}

	return &state, nil
}
