package app

import (
	"fmt"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/configlifecycle"
	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

// App owns runtime configuration plus initialized backend dependencies.
type App struct {
	Config          *config.Config
	Console         *console.Stores
	ConfigLifecycle *configlifecycle.Service
}

// New initializes backend application dependencies from configuration.
func New(cfg *config.Config) (*App, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is required")
	}

	consoleStores, err := console.OpenStore(console.StoreConfig{
		Backend:    console.StoreBackendType(cfg.ConsoleStoreBackend),
		SQLitePath: cfg.ConsoleDBPath,
		DSN:        cfg.ConsoleStoreDSN,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize console stores: %w", err)
	}

	return &App{
		Config:          cfg,
		Console:         consoleStores,
		ConfigLifecycle: configlifecycle.NewWithStores(cfg.AbsConfigPath, cfg.ConfigDir, consoleStores),
	}, nil
}

// Close releases initialized application dependencies.
func (a *App) Close() error {
	if a == nil || a.Console == nil || a.Console.Lifecycle == nil {
		return nil
	}
	return a.Console.Lifecycle.Close()
}
