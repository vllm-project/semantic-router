package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func configureRouterLearningStateStore(cfg *config.RouterConfig) {
	if cfg == nil {
		return
	}
	stateConfig := cfg.RouterLearning.StateStore
	switch strings.TrimSpace(stateConfig.Backend) {
	case "", "local":
		sessiontelemetry.SetRouterSessionStateStore(nil)
	case "redis":
		timeout := time.Duration(stateConfig.TimeoutMS) * time.Millisecond
		ttl := time.Duration(stateConfig.TTLSeconds) * time.Second
		store, err := sessiontelemetry.NewRedisRouterSessionStateStore(
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
			sessiontelemetry.SetRouterSessionStateStore(nil)
			logging.ComponentWarnEvent("extproc", "router_learning_state_store_fail_open", map[string]interface{}{
				"backend": "redis",
				"error":   err.Error(),
			})
			return
		}
		sessiontelemetry.SetRouterSessionStateStore(store)
		logging.ComponentEvent("extproc", "router_learning_state_store_initialized", map[string]interface{}{
			"backend": "redis",
		})
	}
}
