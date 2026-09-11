package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func recordCircuitBreakerSkip(name string) {
	circuitBreakerSkipsTotal.WithLabelValues(name).Inc()
	logging.ComponentEvent("classifier", "circuit_breaker_skip", map[string]interface{}{
		"backend": name,
	})
}

func recordCircuitBreakerFailure(name string) {
	logging.ComponentEvent("classifier", "circuit_breaker_failure", map[string]interface{}{
		"backend": name,
	})
}