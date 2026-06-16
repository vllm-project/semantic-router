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

package looper

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"os"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// SecretEnv is the environment variable used to share a looper authentication
// secret across multiple router replicas. When set, every replica trusts the
// same secret so an internal looper call routed (through Envoy) to a different
// replica is still recognized. When unset, each process generates its own
// random secret, which is correct for single-process and sidecar topologies.
const SecretEnv = "VSR_LOOPER_SECRET"

var (
	secretOnce   sync.Once
	looperSecret string
)

// Secret returns the process looper authentication secret, initializing it on
// first use. The secret is sourced from SecretEnv when set; otherwise a
// cryptographically random 256-bit value is generated in memory. It is never
// written to config or disk. The in-process looper client attaches this value
// to every internal call and extproc validates it before honoring the looper
// fast-path, closing the bypass described in issue #1443.
func Secret() string {
	secretOnce.Do(func() { looperSecret = loadSecret() })
	return looperSecret
}

// loadSecret resolves the looper secret from the environment or generates a
// random one. It is separated from Secret so the resolution logic is unit
// testable without the process-wide sync.Once cache.
func loadSecret() string {
	if env := strings.TrimSpace(os.Getenv(SecretEnv)); env != "" {
		logging.ComponentEvent("looper", "secret_source_env", map[string]interface{}{
			"env": SecretEnv,
		})
		return env
	}

	buf := make([]byte, 32) // 256-bit
	if _, err := rand.Read(buf); err != nil {
		// Fail closed: an empty secret never validates, so looper requests are
		// not honored and every request falls back to the full security
		// pipeline. crypto/rand.Read does not fail on supported platforms.
		logging.ComponentErrorEvent("looper", "secret_generation_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return ""
	}
	logging.ComponentEvent("looper", "secret_generated", map[string]interface{}{
		"source": "crypto/rand",
		"bits":   256,
	})
	return hex.EncodeToString(buf)
}

// ValidateSecret reports whether provided matches the process looper secret
// using a constant-time comparison. An empty provided secret, or an empty
// process secret (generation failure), never validates.
func ValidateSecret(provided string) bool {
	if provided == "" {
		return false
	}
	expected := Secret()
	if expected == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(provided), []byte(expected)) == 1
}
