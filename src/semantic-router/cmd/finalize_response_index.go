package main

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

// errFinalizeRequiresRedis is returned when --finalize-response-index is run
// against a deployment whose Response API is not backed by Redis. The
// conversation index this finalizes exists only in the Redis store, so there
// is nothing to sweep for any other backend and silently exiting 0 would let
// an operator believe a barrier had been established when none had.
var errFinalizeRequiresRedis = errors.New("--finalize-response-index requires response_api.store_backend: redis")

// exitIfFinalizeResponseIndex runs the one-shot conversation index
// finalization sweep and exits, when the operator asked for it.
//
// Deliberately placed before model downloads, tracing, the metrics/API
// listeners and the ExtProc server: this initializes nothing but the Redis
// response store, so it can be run as a short-lived Job against the same
// image and config as the router without paying for (or racing) a full
// router boot. Mirrors exitIfDownloadOnly's shape.
//
// This is an administrative action with irreversible consequences — see
// responsestore.RedisStore.FinalizeConversationIndex for the rollout
// prerequisites (index-aware code deployed everywhere, every index-unaware
// writer drained, Cluster resharding paused, and no rollback afterward)
// that must hold before it is safe to run.
func exitIfFinalizeResponseIndex(finalizeResponseIndex bool, cfg *config.RouterConfig) {
	if !finalizeResponseIndex {
		return
	}

	stats, err := runResponseIndexFinalization(context.Background(), cfg)
	if err != nil {
		logging.ComponentFatalEvent("router", "finalize_response_index_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	logging.ComponentEvent("router", "finalize_response_index_complete", map[string]interface{}{
		"responses_scanned": stats.ResponsesScanned,
		"responses_indexed": stats.ResponsesIndexed,
	})
	fmt.Fprintf(os.Stdout, "conversation index finalization complete: responses_scanned=%d responses_indexed=%d\n",
		stats.ResponsesScanned, stats.ResponsesIndexed)
	os.Exit(0)
}

// runResponseIndexFinalization opens the configured Redis response store,
// finalizes its conversation index, and closes the store again. Split out
// from exitIfFinalizeResponseIndex so the work is testable without the
// os.Exit.
func runResponseIndexFinalization(
	ctx context.Context,
	cfg *config.RouterConfig,
) (responsestore.ConversationIndexFinalizationStats, error) {
	var stats responsestore.ConversationIndexFinalizationStats

	store, err := newResponseIndexStore(cfg)
	if err != nil {
		return stats, err
	}
	defer func() {
		if closeErr := store.Close(); closeErr != nil {
			logging.ComponentWarnEvent("router", "finalize_response_index_store_close_failed", map[string]interface{}{
				"error": closeErr.Error(),
			})
		}
	}()

	return store.FinalizeConversationIndex(ctx)
}

// newResponseIndexStore builds only the Redis response store this command
// needs, from the same response_api config the router itself uses.
func newResponseIndexStore(cfg *config.RouterConfig) (*responsestore.RedisStore, error) {
	if cfg == nil {
		return nil, errors.New("router configuration is unavailable")
	}
	if responsestore.StoreBackendType(cfg.ResponseAPI.StoreBackend) != responsestore.RedisStoreType {
		return nil, fmt.Errorf("%w (configured backend: %q)", errFinalizeRequiresRedis, cfg.ResponseAPI.StoreBackend)
	}

	store, err := responsestore.NewRedisStore(responsestore.StoreConfig{
		Enabled:     true,
		TTLSeconds:  cfg.ResponseAPI.TTLSeconds,
		BackendType: responsestore.RedisStoreType,
		Redis: responsestore.RedisStoreConfig{
			Address:          cfg.ResponseAPI.Redis.Address,
			Password:         cfg.ResponseAPI.Redis.Password,
			DB:               cfg.ResponseAPI.Redis.DB,
			KeyPrefix:        cfg.ResponseAPI.Redis.KeyPrefix,
			ClusterMode:      cfg.ResponseAPI.Redis.ClusterMode,
			ClusterAddresses: cfg.ResponseAPI.Redis.ClusterAddresses,
			PoolSize:         cfg.ResponseAPI.Redis.PoolSize,
			MinIdleConns:     cfg.ResponseAPI.Redis.MinIdleConns,
			MaxRetries:       cfg.ResponseAPI.Redis.MaxRetries,
			DialTimeout:      cfg.ResponseAPI.Redis.DialTimeout,
			ReadTimeout:      cfg.ResponseAPI.Redis.ReadTimeout,
			WriteTimeout:     cfg.ResponseAPI.Redis.WriteTimeout,
			TLSEnabled:       cfg.ResponseAPI.Redis.TLSEnabled,
			TLSCertPath:      cfg.ResponseAPI.Redis.TLSCertPath,
			TLSKeyPath:       cfg.ResponseAPI.Redis.TLSKeyPath,
			TLSCAPath:        cfg.ResponseAPI.Redis.TLSCAPath,
			ConfigPath:       cfg.ResponseAPI.Redis.ConfigPath,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Redis response store: %w", err)
	}

	return store, nil
}
