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

package selection

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// defaultAutoSaveInterval is the fallback cadence for Elo/RL rating persistence
// used whenever a configured interval is missing or invalid, so a misconfigured
// value can neither panic time.NewTicker nor silently disable periodic saves.
const defaultAutoSaveInterval = 30 * time.Second

// resolveAutoSaveInterval turns a raw auto_save_interval config string into a
// safe, strictly-positive duration. It delegates validation to
// config.ParsePeriodicInterval (positive and below the documented maximum) and
// falls back to defaultAutoSaveInterval with a warning on any invalid value.
func resolveAutoSaveInterval(raw string) time.Duration {
	interval, err := config.ParsePeriodicInterval(raw, defaultAutoSaveInterval)
	if err != nil {
		logging.Warnf("[selection] invalid auto_save_interval %q: %v; using default %s", raw, err, defaultAutoSaveInterval)
		return defaultAutoSaveInterval
	}
	return interval
}
