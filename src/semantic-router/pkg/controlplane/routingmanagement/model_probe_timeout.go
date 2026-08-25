package routingmanagement

import (
	"fmt"
	"time"
)

const maximumModelProbeTimeout = 5 * time.Minute

// resolveModelProbeTimeout keeps the bounded control-plane health check
// independent from a Model's potentially long inference deadline. An explicit
// timeout is an operator input and must already fit the probe bound; the stored
// request timeout is only a default and is capped for the health check.
func resolveModelProbeTimeout(requestTimeout string, explicit time.Duration) (time.Duration, error) {
	if explicit != 0 {
		if explicit < time.Second || explicit > maximumModelProbeTimeout {
			return 0, fmt.Errorf("%w: probe timeout must be between 1s and 5m", ErrInvalid)
		}
		return explicit, nil
	}
	configured, err := time.ParseDuration(requestTimeout)
	if err != nil || configured < time.Second {
		return 0, fmt.Errorf("%w: stored Model timeout is invalid", ErrProbeUnavailable)
	}
	if configured > maximumModelProbeTimeout {
		return maximumModelProbeTimeout, nil
	}
	return configured, nil
}
