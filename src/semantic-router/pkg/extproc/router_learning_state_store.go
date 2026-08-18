package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

var newRouterSessionStateStore = sessiontelemetry.NewRedisRouterSessionStateStore

func buildRouterLearningStateStore(cfg *config.RouterConfig) *sessiontelemetry.RouterSessionStateStoreSlot {
	if cfg == nil {
		return nil
	}
	stateConfig := cfg.RouterLearning.StateStore
	switch strings.TrimSpace(stateConfig.Backend) {
	case "", "local":
		return nil
	case "redis":
		timeout := time.Duration(stateConfig.TimeoutMS) * time.Millisecond
		ttl := time.Duration(stateConfig.TTLSeconds) * time.Second
		store, err := newRouterSessionStateStore(
			sessiontelemetry.RedisRouterSessionStoreConfig{
				Address:   stateConfig.Redis.Address,
				Password:  stateConfig.Redis.Password,
				Database:  stateConfig.Redis.Database,
				Timeout:   timeout,
				TTL:       ttl,
				KeyPrefix: stateConfig.Redis.KeyPrefix,
			},
		)
		if err != nil {
			logging.ComponentWarnEvent("extproc", "router_learning_state_store_fail_open", map[string]interface{}{
				"backend": "redis",
				"error":   err.Error(),
			})
			return nil
		}
		return sessiontelemetry.NewRouterSessionStateStoreSlot(store)
	}
	return nil
}

func publishRouterLearningStateStore(router *OpenAIRouter) {
	if router == nil {
		return
	}
	sessiontelemetry.PublishRouterSessionStateStore(router.routerSessionStateStore)
	if router.routerSessionStateStore != nil {
		logging.ComponentEvent("extproc", "router_learning_state_store_initialized", map[string]interface{}{
			"backend": "redis",
		})
	}
}
