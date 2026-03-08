package app

import (
	"fmt"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/configlifecycle"
	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

// App owns runtime configuration plus initialized backend dependencies.
type App struct {
	Config          *config.Config
	Console         *console.Stores
	Auth            *auth.Service
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

	authService, err := auth.New(auth.Config{
		Mode:              auth.Mode(cfg.AuthMode),
		SessionCookieName: cfg.AuthSessionCookieName,
		SessionTTL:        cfg.AuthSessionTTL,
		BootstrapUserID:   cfg.AuthBootstrapUserID,
		BootstrapEmail:    cfg.AuthBootstrapEmail,
		BootstrapName:     cfg.AuthBootstrapName,
		BootstrapRole:     console.ConsoleRole(cfg.AuthBootstrapRole),
		BootstrapSubject:  cfg.AuthBootstrapSubject,
		ProxyUserHeader:   cfg.AuthProxyUserHeader,
		ProxyEmailHeader:  cfg.AuthProxyEmailHeader,
		ProxyNameHeader:   cfg.AuthProxyNameHeader,
		ProxyRolesHeader:  cfg.AuthProxyRolesHeader,
	}, consoleStores)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize dashboard auth service: %w", err)
	}

	return &App{
		Config:          cfg,
		Console:         consoleStores,
		Auth:            authService,
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
