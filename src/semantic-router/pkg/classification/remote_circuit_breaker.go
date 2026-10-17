package classification

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
)

func classifierCircuitBreaker(cfg *config.ExternalModelConfig, operation string) *connector.CircuitBreakerOptions {
	failures, interval := cfg.CircuitBreaker.EffectiveValues()
	if failures == 0 && interval == 0 {
		return nil
	}
	name := cfg.Name
	if name == "" {
		name = operation
	}
	return &connector.CircuitBreakerOptions{Name: name, ConsecutiveFailures: failures, OpenInterval: time.Duration(interval) * time.Millisecond}
}

func classifierCallContext(ctx context.Context, timeout time.Duration, client *connector.Client) (context.Context, context.CancelFunc) {
	if client.CircuitBreakerEnabled() {
		ctx = connector.WithCallerContext(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}
