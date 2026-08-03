package backend

import (
	"context"
	"errors"
	"fmt"
	"time"
)

const DefaultCollectionInterval = 2 * time.Second

// RunnerErrorHandler receives non-fatal adapter collection errors.
type RunnerErrorHandler func(kind EngineKind, err error)

// Runner periodically collects telemetry from adapters and writes normalized
// samples into the configured store.
type Runner struct {
	adapters []TelemetryAdapter
	store    *Store
	interval time.Duration
	onError  RunnerErrorHandler
}

// RunnerConfig configures a backend telemetry runner.
type RunnerConfig struct {
	Adapters []TelemetryAdapter
	Store    *Store
	Interval time.Duration
	OnError  RunnerErrorHandler
}

// NewRunner creates a telemetry runner. Adapter failures are non-fatal during
// collection; invalid runner wiring is reported here.
func NewRunner(cfg RunnerConfig) (*Runner, error) {
	if len(cfg.Adapters) == 0 {
		return nil, fmt.Errorf("backend telemetry runner requires at least one adapter")
	}
	for index, adapter := range cfg.Adapters {
		if adapter == nil {
			return nil, fmt.Errorf("backend telemetry runner adapter %d is nil", index)
		}
	}
	if cfg.Store == nil {
		cfg.Store = defaultStore
	}
	if cfg.Interval <= 0 {
		cfg.Interval = DefaultCollectionInterval
	}
	return &Runner{
		adapters: append([]TelemetryAdapter(nil), cfg.Adapters...),
		store:    cfg.Store,
		interval: cfg.Interval,
		onError:  cfg.OnError,
	}, nil
}

// CollectOnce runs one collection pass across all adapters.
func (r *Runner) CollectOnce(ctx context.Context) error {
	if r == nil {
		return fmt.Errorf("backend telemetry runner is nil")
	}
	var errs []error
	for _, adapter := range r.adapters {
		samples, err := adapter.Collect(ctx)
		if len(samples) > 0 {
			if upsertErr := r.store.UpsertMany(samples); upsertErr != nil {
				r.reportError(adapter.EngineKind(), upsertErr)
				errs = append(errs, fmt.Errorf("%s: %w", adapter.EngineKind(), upsertErr))
			}
		}
		if err != nil {
			r.reportError(adapter.EngineKind(), err)
			errs = append(errs, fmt.Errorf("%s: %w", adapter.EngineKind(), err))
			continue
		}
	}
	return errors.Join(errs...)
}

// Run collects immediately, then repeats until ctx is canceled.
func (r *Runner) Run(ctx context.Context) {
	if r == nil {
		return
	}
	_ = r.CollectOnce(ctx)

	ticker := time.NewTicker(r.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			_ = r.CollectOnce(ctx)
		}
	}
}

func (r *Runner) reportError(kind EngineKind, err error) {
	if r.onError != nil && err != nil {
		r.onError(kind, err)
	}
}
