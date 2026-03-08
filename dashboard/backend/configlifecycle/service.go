package configlifecycle

import (
	"path/filepath"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

// Service owns file-backed config lifecycle operations for the dashboard.
type Service struct {
	ConfigPath string
	ConfigDir  string
	Stores     *console.Stores
}

// New creates a lifecycle service for the given config file workspace.
func New(configPath, configDir string) *Service {
	return NewWithStores(configPath, configDir, nil)
}

// NewWithStores creates a lifecycle service with optional console persistence.
func NewWithStores(configPath, configDir string, stores *console.Stores) *Service {
	if configDir == "" {
		configDir = filepath.Dir(configPath)
	}
	return &Service{
		ConfigPath: configPath,
		ConfigDir:  configDir,
		Stores:     stores,
	}
}
