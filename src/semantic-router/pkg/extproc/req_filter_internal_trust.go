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

package extproc

import (
	"crypto/subtle"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// enforceInternalHeaderTrust establishes the trust boundary for the router's
// internal looper leg at ingress, before any router logic keys off the looper
// markers.
//
// A genuine looper re-dispatch authenticates with a per-process secret
// (headers.VSRLooperAuthorization). On such a trusted request the looper marker
// is honored and the proof is dropped so it never travels upstream. On any
// untrusted (client-facing) request every reserved internal header is stripped
// from ctx.Headers so a caller cannot spoof the internal looper path — which
// would skip guardrail plugins — or inject a caller-identity carrier
// (x-vsr-inbound-authorization) that a forward_authorization_header backend
// would otherwise trust as the caller's credential.
func enforceInternalHeaderTrust(ctx *RequestContext) {
	if ctx == nil || ctx.Headers == nil {
		return
	}

	if isTrustedInternalRequest(ctx.Headers) {
		marker := lookupHeaderCaseInsensitive(ctx.Headers, headers.VSRLooperRequest)
		if strings.EqualFold(strings.TrimSpace(marker), "true") {
			ctx.LooperRequest = true
		}
		// The proof has served its purpose; drop it so it never propagates.
		deleteHeaderCaseInsensitive(ctx.Headers, headers.VSRLooperAuthorization)
		return
	}

	// Untrusted ingress: the looper marker (if any) was spoofed. Strip every
	// reserved internal header and never treat the request as internal.
	stripped := false
	for _, name := range headers.ReservedInternalHeaders {
		if deleteHeaderCaseInsensitive(ctx.Headers, name) {
			stripped = true
		}
	}
	ctx.LooperRequest = false

	if stripped {
		logging.ComponentWarnEvent("extproc", "reserved_internal_headers_stripped", map[string]interface{}{
			"request_id": ctx.RequestID,
			"detail":     "client-supplied reserved internal headers were stripped at ingress",
		})
	}
}

// isTrustedInternalRequest reports whether the request carries a valid
// internal-leg auth proof. The comparison is constant-time; an empty or absent
// proof (or an unset process secret) is never trusted.
func isTrustedInternalRequest(h map[string]string) bool {
	secret := looper.InternalAuthSecret()
	if secret == "" {
		return false
	}
	provided := strings.TrimSpace(lookupHeaderCaseInsensitive(h, headers.VSRLooperAuthorization))
	if provided == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(provided), []byte(secret)) == 1
}

// deleteHeaderCaseInsensitive removes every entry whose key case-insensitively
// matches name and reports whether anything was removed. Inbound header keys are
// captured verbatim (HTTP/1.1 casing can vary), so a client could otherwise vary
// the casing of a reserved header to slip past a plain map delete.
func deleteHeaderCaseInsensitive(h map[string]string, name string) bool {
	removed := false
	if _, ok := h[name]; ok {
		delete(h, name)
		removed = true
	}
	for k := range h {
		if strings.EqualFold(k, name) {
			delete(h, k)
			removed = true
		}
	}
	return removed
}
