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
	"sync"
)

var (
	internalAuthOnce   sync.Once
	internalAuthSecret string
)

// InternalAuthSecret returns the per-process secret that authenticates the
// internal looper leg.
//
// The looper re-dispatch loops back to the same router process, so the client
// (this package) and the extproc that validates the request share memory. The
// client stamps the secret on every internal request via the
// headers.VSRLooperAuthorization header, and extproc validates it at the
// trusted ingress before honoring any looper marker or caller-identity carrier.
// Because the secret is generated once, kept only in memory, and never leaves
// the process, a client on the wire cannot forge it.
//
// The secret is unpredictable (crypto/rand) and non-empty. Generation failure
// would silently disable the internal-leg authentication, collapsing the trust
// boundary, so it panics rather than returning an empty secret.
func InternalAuthSecret() string {
	internalAuthOnce.Do(func() {
		buf := make([]byte, 32)
		if _, err := rand.Read(buf); err != nil {
			panic("looper: failed to generate internal auth secret: " + err.Error())
		}
		internalAuthSecret = hex.EncodeToString(buf)
	})
	return internalAuthSecret
}
