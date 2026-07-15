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
	"encoding/hex"
	"os"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// minInternalAuthSecretLen is the shortest operator-provided secret we accept
// without warning. A short shared secret is guessable, which would let a client
// forge the internal-leg proof. The random fallback is well above this.
const minInternalAuthSecretLen = 16

// InternalAuthSecretEnv names an optional operator-provided secret that
// authenticates the internal looper leg. Set it to the SAME value on every
// router replica when the looper endpoint is a shared/load-balanced address
// (so a re-dispatch may land on a different pod than the one that issued it);
// the per-process random fallback only holds when the leg loops back to the
// same process (the default localhost sidecar topology).
const InternalAuthSecretEnv = "VSR_LOOPER_INTERNAL_AUTH_SECRET" //nolint:gosec // G101: this is the NAME of an env var, not a hardcoded credential.

var (
	internalAuthOnce   sync.Once
	internalAuthSecret string
)

// InternalAuthSecret returns the secret that authenticates the internal looper
// leg. The looper client stamps it on every internal request via the
// headers.VSRLooperAuthorization header, and extproc validates it at the
// trusted ingress before honoring any looper marker or caller-identity carrier.
//
// It resolves once, in order:
//   - VSR_LOOPER_INTERNAL_AUTH_SECRET, when set — supports multi-replica
//     topologies where the re-dispatch may land on a different process.
//   - otherwise a per-process value from crypto/rand — correct for the default
//     localhost sidecar topology where the leg loops back to the same process.
//
// The secret is stamped on the internal request and travels the loopback wire,
// but extproc validates and strips it at ingress so it never reaches an upstream.
//
// The result is always non-empty and unpredictable. A crypto/rand failure would
// silently disable authentication and collapse the trust boundary, so it panics
// rather than returning an empty secret.
func InternalAuthSecret() string {
	internalAuthOnce.Do(func() {
		if env := strings.TrimSpace(os.Getenv(InternalAuthSecretEnv)); env != "" {
			if len(env) < minInternalAuthSecretLen {
				logging.ComponentWarnEvent("looper", "internal_auth_secret_weak", map[string]interface{}{
					"env":        InternalAuthSecretEnv,
					"length":     len(env),
					"min_length": minInternalAuthSecretLen,
					"detail":     "operator-provided internal auth secret is short and may be guessable; use a long random value",
				})
			}
			internalAuthSecret = env
			return
		}
		buf := make([]byte, 32)
		if _, err := rand.Read(buf); err != nil {
			panic("looper: failed to generate internal auth secret: " + err.Error())
		}
		internalAuthSecret = hex.EncodeToString(buf)
	})
	return internalAuthSecret
}
