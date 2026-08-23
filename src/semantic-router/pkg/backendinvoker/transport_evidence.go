package backendinvoker

import (
	"errors"
	"fmt"
)

// knownZeroTransportEvidence is deliberately package-sealed. A transport may
// obtain one through NewKnownZeroTransportFailure only after it has proved
// that no request or response bytes crossed the provider boundary. Ordinary
// network errors carry no such authority.
type knownZeroTransportEvidence interface {
	error
	knownZeroFallbackTrigger() FallbackTrigger
}

type knownZeroTransportFailure struct {
	cause   error
	trigger FallbackTrigger
}

func (failure *knownZeroTransportFailure) Error() string { return failure.cause.Error() }

func (failure *knownZeroTransportFailure) Unwrap() error { return failure.cause }

func (failure *knownZeroTransportFailure) knownZeroFallbackTrigger() FallbackTrigger {
	return failure.trigger
}

// NewKnownZeroTransportFailure creates explicit, closed-vocabulary transport
// evidence. Invalid evidence degrades to an ordinary error and therefore can
// never authorize a retry or cross-Model fallback.
func NewKnownZeroTransportFailure(trigger FallbackTrigger, cause error) error {
	if cause == nil {
		cause = errors.New("transport failed before dispatch")
	}
	if fallbackTriggerOrder(trigger) < 0 {
		return fmt.Errorf("invalid known-zero fallback trigger %q: %w", trigger, cause)
	}
	return &knownZeroTransportFailure{cause: cause, trigger: trigger}
}

func knownZeroTrigger(err error) (FallbackTrigger, bool) {
	var evidence knownZeroTransportEvidence
	if !errors.As(err, &evidence) {
		return "", false
	}
	trigger := evidence.knownZeroFallbackTrigger()
	return trigger, fallbackTriggerOrder(trigger) >= 0
}
