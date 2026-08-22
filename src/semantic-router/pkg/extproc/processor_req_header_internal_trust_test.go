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
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/internalauth"
)

// TestAuthenticateLooperRequestContextStripsSpoofedMarkers proves a client
// cannot forge the internal looper path or a caller-identity carrier by sending
// the reserved headers without the process-local internal credential.
func TestAuthenticateLooperRequestContextStripsSpoofedMarkers(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		"authorization":                 "Bearer real-caller-key",
		headers.VSRLooperRequest:        "true",
		headers.VSRLooperDecision:       "attacker_decision",
		headers.VSRLooperIteration:      "3",
		headers.VSRInboundAuthorization: "Bearer victim-virtual-key",
		headers.VSRFusionDepth:          "1",
	}}

	authenticateLooperRequestContext(ctx)

	assert.False(t, ctx.LooperRequest,
		"a spoofed looper marker without the internal credential must not be honored")
	for _, name := range headers.ReservedInternalHeaders {
		_, present := ctx.Headers[name]
		assert.Falsef(t, present, "reserved header %q must be stripped on an untrusted request", name)
	}
	// The caller's own Authorization is untouched — only reserved internal
	// headers are stripped at this boundary.
	assert.Equal(t, "Bearer real-caller-key", ctx.Headers["authorization"])
}

// TestAuthenticateLooperRequestContextStripsMixedCaseSpoof proves the strip is
// case-insensitive, so a client cannot evade it by varying header casing.
func TestAuthenticateLooperRequestContextStripsMixedCaseSpoof(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		"X-VSR-Looper-Request":        "true",
		"X-VSR-Inbound-Authorization": "Bearer victim-virtual-key",
	}}

	authenticateLooperRequestContext(ctx)

	assert.False(t, ctx.LooperRequest)
	assert.Empty(t, ctx.Headers, "mixed-case reserved headers must be stripped too")
}

// TestAuthenticateLooperRequestContextTrustsAuthenticatedLeg proves a genuine
// internal re-dispatch keeps the looper marker and the caller-identity carrier
// that forward_authorization_header reads, while the credential itself is
// consumed so it never travels further.
func TestAuthenticateLooperRequestContextTrustsAuthenticatedLeg(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		headers.VSRInternalAuth:         internalauth.Token(),
		headers.VSRLooperRequest:        "true",
		headers.VSRLooperDecision:       "genuine_decision",
		headers.VSRInboundAuthorization: "Bearer user-virtual-key",
	}}

	authenticateLooperRequestContext(ctx)

	assert.True(t, ctx.LooperRequest, "an authenticated internal leg must be honored")
	assert.Equal(t, "Bearer user-virtual-key", ctx.Headers[headers.VSRInboundAuthorization],
		"the caller-identity carrier must survive on a trusted leg")
	assert.Equal(t, "genuine_decision", ctx.Headers[headers.VSRLooperDecision])
	_, present := ctx.Headers[headers.VSRInternalAuth]
	assert.False(t, present, "the internal credential must be consumed at ingress")
}

// TestAuthenticateLooperRequestContextRejectsWrongCredential proves a near-miss
// credential is treated exactly like a spoof.
func TestAuthenticateLooperRequestContextRejectsWrongCredential(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		headers.VSRInternalAuth:         internalauth.Token() + "x",
		headers.VSRLooperRequest:        "true",
		headers.VSRInboundAuthorization: "Bearer victim-virtual-key",
	}}

	authenticateLooperRequestContext(ctx)

	assert.False(t, ctx.LooperRequest)
	assert.Empty(t, ctx.Headers)
}

// TestLooperInternalHeadersForRemovalCoversReservedSet proves the wire-level
// strip covers every reserved internal header: the in-memory strip above does
// not reach the request Envoy forwards upstream.
func TestLooperInternalHeadersForRemovalCoversReservedSet(t *testing.T) {
	removal := looperInternalHeadersForRemoval()
	for _, name := range headers.ReservedInternalHeaders {
		assert.Containsf(t, removal, name, "reserved header %q must be removed from the wire", name)
	}
}
