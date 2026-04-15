package startupstatus

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
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

// StatusWriter persists startup state so the dashboard (or other consumers)
// can track router startup progress. Implementations include FileWriter
// (local JSON file) and RedisWriter (shared key in Redis).
type StatusWriter interface {
	Write(state State) error
}

var _ StatusWriter = (*FileWriter)(nil)

// FileWriter persists startup state to a JSON file.
type FileWriter struct {
	path string
	mu   sync.Mutex
}

var statusDirCache sync.Map

// NewFileWriter creates a file-based StatusWriter using a router config path.
func NewFileWriter(configPath string) *FileWriter {
	return &FileWriter{path: StatusPathFromConfigPath(configPath)}
}

// StatusPathFromConfigPath returns the runtime status path, preferring the config
// directory when it is writable and otherwise falling back to a temp-owned path.
func StatusPathFromConfigPath(configPath string) string {
	statusDir := runtimeStatusDirFromConfigPath(configPath)
	return filepath.Join(statusDir, "router-runtime.json")
}

func runtimeStatusDirFromConfigPath(configPath string) string {
	if overrideDir := os.Getenv("VLLM_SR_RUNTIME_STATUS_DIR"); overrideDir != "" {
		return overrideDir
	}

	configDir := filepath.Dir(configPath)
	cacheKey := filepath.Clean(configDir)
	if cached, ok := statusDirCache.Load(cacheKey); ok {
		return cached.(string)
	}

	statusDir := filepath.Join(os.TempDir(), "vllm-sr", "runtime-status", stablePathToken(configDir))
	if dirWritable(configDir) {
		statusDir = configDir
	}
	statusDirCache.Store(cacheKey, statusDir)
	return statusDir
}

func dirWritable(path string) bool {
	info, err := os.Stat(path)
	if err != nil || !info.IsDir() {
		return false
	}
	probe, err := os.CreateTemp(path, ".vllm-sr-write-check-*")
	if err != nil {
		return false
	}
	probePath := probe.Name()
	if closeErr := probe.Close(); closeErr != nil {
		_ = os.Remove(probePath)
		return false
	}
	return os.Remove(probePath) == nil
}

func stablePathToken(path string) string {
	hasher := fnv.New64a()
	_, _ = hasher.Write([]byte(filepath.Clean(path)))
	return fmt.Sprintf("%016x", hasher.Sum64())
}

// Write persists the provided state atomically to a JSON file.
func (w *FileWriter) Write(state State) error {
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
