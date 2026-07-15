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

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

// TestEnforceInternalHeaderTrustStripsSpoofedMarkers proves a client cannot
// forge the internal looper path or a caller-identity carrier by sending the
// reserved headers without the per-process auth proof.
func TestEnforceInternalHeaderTrustStripsSpoofedMarkers(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		"authorization":                 "Bearer real-caller-key",
		headers.VSRLooperRequest:        "true",
		headers.VSRLooperDecision:       "attacker_decision",
		headers.VSRLooperIteration:      "3",
		headers.VSRInboundAuthorization: "Bearer victim-virtual-key",
		headers.VSRFusionDepth:          "1",
	}}

	enforceInternalHeaderTrust(ctx)

	assert.False(t, ctx.LooperRequest,
		"a spoofed looper marker without the auth proof must not be honored")
	for _, name := range headers.ReservedInternalHeaders {
		_, present := ctx.Headers[name]
		assert.Falsef(t, present, "reserved header %q must be stripped on an untrusted request", name)
	}
	// The caller's own Authorization is untouched — only reserved internal
	// headers are stripped at this boundary.
	assert.Equal(t, "Bearer real-caller-key", ctx.Headers["authorization"])
}

// TestEnforceInternalHeaderTrustStripsMixedCaseSpoof proves the strip is
// case-insensitive, so a client cannot evade it by varying header casing.
func TestEnforceInternalHeaderTrustStripsMixedCaseSpoof(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		"X-VSR-Looper-Request":        "true",
		"X-Vsr-Inbound-Authorization": "Bearer victim-virtual-key",
	}}

	enforceInternalHeaderTrust(ctx)

	assert.False(t, ctx.LooperRequest)
	assert.Empty(t, ctx.Headers, "mixed-case reserved headers must be stripped")
}

// TestEnforceInternalHeaderTrustHonorsAuthenticatedLeg proves a genuine
// internal request (valid proof) is treated as a looper request, keeps its
// caller-identity carrier, and has the proof stripped so it never propagates.
func TestEnforceInternalHeaderTrustHonorsAuthenticatedLeg(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		headers.VSRLooperRequest:        "true",
		headers.VSRLooperAuthorization:  looper.InternalAuthSecret(),
		headers.VSRInboundAuthorization: "Bearer user-virtual-key",
	}}

	enforceInternalHeaderTrust(ctx)

	assert.True(t, ctx.LooperRequest, "an authenticated internal leg must be honored")
	// The carrier survives for the forward-auth backend to consume.
	assert.Equal(t, "Bearer user-virtual-key", ctx.Headers[headers.VSRInboundAuthorization])
	// The proof itself is consumed and never forwarded.
	_, present := ctx.Headers[headers.VSRLooperAuthorization]
	assert.False(t, present, "the internal auth proof must be stripped after validation")
}

// TestEnforceInternalHeaderTrustRejectsWrongSecret proves a guessed/stale proof
// is rejected and the request is downgraded to untrusted.
func TestEnforceInternalHeaderTrustRejectsWrongSecret(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		headers.VSRLooperRequest:       "true",
		headers.VSRLooperAuthorization: "not-the-real-secret",
	}}

	enforceInternalHeaderTrust(ctx)

	assert.False(t, ctx.LooperRequest, "an invalid auth proof must not be trusted")
	_, present := ctx.Headers[headers.VSRLooperRequest]
	assert.False(t, present, "reserved headers must be stripped when the proof is invalid")
}

// TestHandleRequestHeadersRemovesReservedHeadersFromWire proves the reserved
// internal headers — critically the auth proof — are scheduled for wire removal
// so they never reach an upstream, even on a genuine (authenticated) leg. The
// in-memory strip in enforceInternalHeaderTrust is not enough; Envoy forwards
// the original wire headers unless extproc emits RemoveHeaders.
func TestHandleRequestHeadersRemovesReservedHeadersFromWire(t *testing.T) {
	router := &OpenAIRouter{}
	req := &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":method", Value: "POST"},
					{Key: ":path", Value: "/v1/chat/completions"},
					{Key: "x-vsr-looper-request", Value: "true"},
					{Key: headers.VSRLooperAuthorization, Value: looper.InternalAuthSecret()},
				},
			},
		},
	}
	ctx := &RequestContext{Headers: make(map[string]string)}
	resp, err := router.handleRequestHeaders(req, ctx)
	require.NoError(t, err)

	removed := resp.GetRequestHeaders().GetResponse().GetHeaderMutation().GetRemoveHeaders()
	for _, name := range headers.ReservedInternalHeaders {
		assert.Containsf(t, removed, name, "reserved header %q must be scheduled for wire removal", name)
	}
}

// TestIsTrustedInternalRequest exercises the matcher directly.
func TestIsTrustedInternalRequest(t *testing.T) {
	assert.True(t, isTrustedInternalRequest(map[string]string{
		headers.VSRLooperAuthorization: looper.InternalAuthSecret(),
	}))
	assert.False(t, isTrustedInternalRequest(map[string]string{
		headers.VSRLooperAuthorization: "",
	}))
	assert.False(t, isTrustedInternalRequest(map[string]string{}))
	assert.False(t, isTrustedInternalRequest(map[string]string{
		headers.VSRLooperAuthorization: looper.InternalAuthSecret() + "x",
	}))
}
