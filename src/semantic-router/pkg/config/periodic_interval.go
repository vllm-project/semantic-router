/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package config

import (
	"fmt"
	"strings"
	"time"
)

// MaxPeriodicInterval bounds every ticker-driving interval accepted by the
// router (Elo/RL auto-save flushes and lookup-table auto-save/replay-populate
// cadence). Anything larger is almost certainly a misconfiguration: it would
// make periodic work effectively never run and risks duration overflow. This
// is the documented maximum referenced by the router configuration contract.
const MaxPeriodicInterval = 24 * time.Hour

// ParsePeriodicInterval parses a periodic interval string such as "30s" or
// "5m" into a time.Duration suitable for driving a time.Ticker.
//
// An empty (or whitespace-only) value yields def with no error, letting
// callers keep their documented default. A non-empty value must parse and be
// strictly positive and no greater than MaxPeriodicInterval; otherwise a
// descriptive error is returned (with the offending value quoted) so callers
// can reject the configuration or fall back to a safe default. Because
// time.NewTicker panics on a non-positive duration, callers must never pass a
// value that failed this check to a ticker.
func ParsePeriodicInterval(value string, def time.Duration) (time.Duration, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return def, nil
	}

	d, err := time.ParseDuration(trimmed)
	if err != nil {
		return 0, fmt.Errorf("invalid duration %q: %w", value, err)
	}
	if d <= 0 {
		return 0, fmt.Errorf("duration must be positive, got %q", value)
	}
	if d > MaxPeriodicInterval {
		return 0, fmt.Errorf("duration %q exceeds maximum %s", value, MaxPeriodicInterval)
	}
	return d, nil
}
