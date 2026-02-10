package adapter

import (
	"context"
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/adapter/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/adapter/http"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/router/engine"
)

type Adapter interface {
	Start() error
	Stop() error
	GetEngine() *engine.RouterEngine
}
type Manager struct {
	adapters []Adapter
	wg       sync.WaitGroup
	ctx      context.Context
	cancel   context.CancelFunc
}

func NewManager() *Manager {
	ctx, cancel := context.WithCancel(context.Background())
	return &Manager{
		adapters: make([]Adapter, 0),
		ctx:      ctx,
		cancel:   cancel,
	}
}

func (m *Manager) CreateAdapters(cfg *config.RouterConfig, eng *engine.RouterEngine, configPath string) error {
	if err := cfg.ValidateAdapters(); err != nil {
		return fmt.Errorf("adapter validation failed: %w", err)
	}

	enabled := cfg.GetEnabledAdapters()
	if len(enabled) == 0 {
		return fmt.Errorf("no adapters enabled in configuration")
	}

	logging.Infof("Creating %d enabled adapter(s)", len(enabled))

	for _, adapterCfg := range enabled {
		var adapter Adapter
		var err error

		switch adapterCfg.Type {
		case "envoy":
			adapter, err = extproc.NewAdapter(eng, configPath, adapterCfg.Port, adapterCfg.TLS)
			if err != nil {
				return fmt.Errorf("failed to create ExtProc adapter: %w", err)
			}
			if adapterCfg.TLS != nil && adapterCfg.TLS.Enabled {
				logging.Infof("Created ExtProc adapter on port %d (TLS enabled)", adapterCfg.Port)
			} else {
				logging.Infof("Created ExtProc adapter on port %d", adapterCfg.Port)
			}

		case "http":
			adapter, err = http.NewAdapter(eng, adapterCfg.Port)
			if err != nil {
				return fmt.Errorf("failed to create HTTP adapter: %w", err)
			}
			logging.Infof("Created HTTP adapter on port %d", adapterCfg.Port)
		// TODO: Add ngninx adapter and grpc adapter
		default:
			return fmt.Errorf("unknown adapter type: %s", adapterCfg.Type)
		}

		m.adapters = append(m.adapters, adapter)
	}

	return nil
}

// StartAll starts all registered adapters in separate goroutines
func (m *Manager) StartAll() error {
	if len(m.adapters) == 0 {
		return fmt.Errorf("no adapters to start")
	}

	logging.Infof("Starting %d adapter(s)", len(m.adapters))

	for i, adapter := range m.adapters {
		adapterIdx := i
		currentAdapter := adapter

		m.wg.Add(1)
		go func() {
			defer m.wg.Done()

			logging.Infof("Adapter %d starting...", adapterIdx)
			if err := currentAdapter.Start(); err != nil {
				logging.Errorf("Adapter %d error: %v", adapterIdx, err)
			}
			logging.Infof("Adapter %d stopped", adapterIdx)
		}()
	}

	return nil
}

// Wait waits for all adapters to stop
func (m *Manager) Wait() {
	m.wg.Wait()
}

// StopAll stops all adapters gracefully
func (m *Manager) StopAll() error {
	logging.Infof("Stopping all adapters...")

	m.cancel()

	var errs []error
	for i, adapter := range m.adapters {
		if err := adapter.Stop(); err != nil {
			logging.Errorf("Error stopping adapter %d: %v", i, err)
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("failed to stop %d adapter(s)", len(errs))
	}

	logging.Infof("All adapters stopped successfully")
	return nil
}

// GetAdapters returns all registered adapters
func (m *Manager) GetAdapters() []Adapter {
	return m.adapters
}
