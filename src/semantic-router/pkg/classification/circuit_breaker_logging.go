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

// recordCircuitBreakerTransition emits a structured log each time the circuit
// breaker state machine changes state. The backend label stays bounded to the
// finite set of configured remote backends.
func recordCircuitBreakerTransition(name string, from, to circuitBreakerState) {
	logging.ComponentEvent("classifier", "circuit_breaker_transition", map[string]interface{}{
		"backend": name,
		"from":    from.String(),
		"to":      to.String(),
	})
}
