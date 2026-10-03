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
	"encoding/base64"
	"fmt"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/handoff"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

var handoffTestNow = time.Date(2026, time.September, 4, 12, 0, 0, 0, time.UTC)

// encodedHandoffForTest encodes a valid envelope for handoff-test. extra is
// appended verbatim inside the JSON object.
func encodedHandoffForTest(extra string) string {
	return encodedHandoffWithID("handoff-test", extra)
}

func encodedHandoffWithID(id, extra string) string {
	payload := fmt.Sprintf(
		`{"version":"1","handoff_id":%q,"root_invocation_id":"root-test","expires_at":"2026-09-04T12:10:00Z"%s}`,
		id, extra,
	)
	return base64.RawURLEncoding.EncodeToString([]byte(payload))
}

func handoffHeaderRequest(method, path string, values ...string) *ext_proc.ProcessingRequest_RequestHeaders {
	requestHeaders := []*core.HeaderValue{
		{Key: ":method", Value: method},
		{Key: ":path", Value: path},
	}
	for _, value := range values {
		requestHeaders = append(requestHeaders, &core.HeaderValue{
			Key:   headers.VSRHandoffEnvelope,
			Value: value,
		})
	}
	return &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: requestHeaders},
		},
	}
}

func handoffTransportRouter(enabled bool) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{RouterOptions: config.RouterOptions{
			Handoff: config.HandoffConfig{Enabled: enabled},
		}},
		handoffNow:    func() time.Time { return handoffTestNow },
		handoffLedger: handoff.NewLedger(16),
	}
}

func sendHandoffHeaders(
	t *testing.T,
	router *OpenAIRouter,
	request *ext_proc.ProcessingRequest_RequestHeaders,
) (*ext_proc.ProcessingResponse, *RequestContext) {
	t.Helper()
	ctx := &RequestContext{Headers: map[string]string{}}
	response, err := router.handleRequestHeaders(request, ctx)
	require.NoError(t, err)
	assertHandoffHeaderNotCaptured(t, ctx)
	return response, ctx
}

func TestHandoffWithoutEnvelopeLeavesRequestUnchanged(t *testing.T) {
	response, ctx := sendHandoffHeaders(t, handoffTransportRouter(true), handoffHeaderRequest("POST", "/v1/chat/completions"))

	require.NotNil(t, response.GetRequestHeaders())
	assert.Equal(t, handoffReceipt{}, ctx.HandoffReceipt)
}

func TestHandoffAdmissionReportsAcceptedOrPartial(t *testing.T) {
	tests := []struct {
		name       string
		extra      string
		wantStatus string
		wantReason string
	}{
		{name: "identity only", wantStatus: handoffStatusAccepted, wantReason: handoffReasonAdmitted},
		{name: "opaque runtime state", extra: `,"runtime":{"task_summary":"t","tool_state_refs":["ref-1"]}`, wantStatus: handoffStatusAccepted, wantReason: handoffReasonAdmitted},
		{name: "selection facts", extra: `,"selection":{"required_capabilities":["tools"],"remaining_tokens":4096}`, wantStatus: handoffStatusPartial, wantReason: handoffReasonSelectionPending},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := handoffHeaderRequest("POST", "/v1/chat/completions", encodedHandoffForTest(test.extra))
			request.RequestHeaders.Headers.Headers[2].Key = "X-VSR-Handoff-Envelope"

			response, ctx := sendHandoffHeaders(t, handoffTransportRouter(true), request)

			require.NotNil(t, response.GetRequestHeaders())
			assert.Contains(t, response.GetRequestHeaders().GetResponse().GetHeaderMutation().GetRemoveHeaders(), headers.VSRHandoffEnvelope)
			assert.Equal(t, handoffReceipt{Version: "1", ID: "handoff-test", Status: test.wantStatus, Reason: test.wantReason}, ctx.HandoffReceipt)
		})
	}
}

func TestHandoffIgnoredRequestsAreStrippedAndRoutedNormally(t *testing.T) {
	tests := []struct {
		name       string
		enabled    bool
		method     string
		path       string
		skip       bool
		wantReason string
	}{
		{name: "disabled", method: "POST", path: "/v1/chat/completions", wantReason: "feature_disabled"},
		{name: "unsupported method", enabled: true, method: "GET", path: "/v1/chat/completions", wantReason: "unsupported_endpoint"},
		{name: "unsupported path", enabled: true, method: "POST", path: "/v1/embeddings", wantReason: "unsupported_endpoint"},
		{name: "skip processing", enabled: true, method: "POST", path: "/v1/chat/completions", skip: true, wantReason: "skip_processing"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: map[string]string{}, SkipProcessing: test.skip}
			response, err := handoffTransportRouter(test.enabled).handleRequestHeaders(
				handoffHeaderRequest(test.method, test.path, encodedHandoffForTest("")), ctx,
			)

			require.NoError(t, err)
			assertHandoffHeaderNotCaptured(t, ctx)
			if requestHeaders := response.GetRequestHeaders(); requestHeaders != nil {
				assert.Contains(t, requestHeaders.GetResponse().GetHeaderMutation().GetRemoveHeaders(), headers.VSRHandoffEnvelope)
			} else {
				require.NotNil(t, response.GetImmediateResponse(), "baseline endpoint handling must remain intact")
			}
			assert.Equal(t, handoffReceipt{Status: handoffStatusIgnored, Reason: test.wantReason}, ctx.HandoffReceipt)
		})
	}
}

func TestHandoffIsAdmittedOnMaintainedInferenceEndpoints(t *testing.T) {
	for _, path := range []string{"/v1/chat/completions?trace=1", "/v1/responses", "/v1/messages"} {
		t.Run(path, func(t *testing.T) {
			_, ctx := sendHandoffHeaders(t, handoffTransportRouter(true), handoffHeaderRequest("POST", path, encodedHandoffForTest("")))

			assert.Equal(t, handoffStatusAccepted, ctx.HandoffReceipt.Status)
		})
	}
}

func TestHandoffInvalidEnvelopesAreRejectedWithContentFreeReasons(t *testing.T) {
	tests := []struct {
		name       string
		values     []string
		wantCode   int
		wantStatus string
		wantReason string
	}{
		{name: "malformed base64", values: []string{"not+base64"}, wantCode: 400, wantStatus: handoffStatusRejected, wantReason: "malformed_encoding"},
		{name: "encoded too large", values: []string{strings.Repeat("a", maxHandoffHeaderBytes+1)}, wantCode: 413, wantStatus: handoffStatusRejected, wantReason: "payload_too_large"},
		{name: "decoded too large", values: []string{base64.RawURLEncoding.EncodeToString([]byte(strings.Repeat(" ", handoff.MaxJSONBytes+1)))}, wantCode: 413, wantStatus: handoffStatusRejected, wantReason: "payload_too_large"},
		{name: "multiple headers", values: []string{encodedHandoffForTest(""), encodedHandoffForTest("")}, wantCode: 400, wantStatus: handoffStatusRejected, wantReason: "multiple_headers"},
		{name: "unknown field", values: []string{encodedHandoffForTest(`,"transcript":"x"`)}, wantCode: 400, wantStatus: handoffStatusRejected, wantReason: "unknown_field"},
		{name: "expired", values: []string{base64.RawURLEncoding.EncodeToString([]byte(`{"version":"1","handoff_id":"h","root_invocation_id":"r","expires_at":"2026-09-04T11:59:00Z"}`))}, wantCode: 422, wantStatus: handoffStatusExpired, wantReason: "expired"},
		{name: "newer version", values: []string{base64.RawURLEncoding.EncodeToString([]byte(`{"version":"2"}`))}, wantCode: 400, wantStatus: handoffStatusRejected, wantReason: "unsupported_version"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response, ctx := sendHandoffHeaders(t, handoffTransportRouter(true), handoffHeaderRequest("POST", "/v1/chat/completions", test.values...))

			require.NotNil(t, response.GetImmediateResponse())
			assert.Equal(t, test.wantCode, int(response.GetImmediateResponse().GetStatus().GetCode()))
			assert.Equal(t, handoffReceipt{Status: test.wantStatus, Reason: test.wantReason}, ctx.HandoffReceipt)
		})
	}
}

func TestHandoffRetriesAreIdempotentAndIDReuseConflicts(t *testing.T) {
	router := handoffTransportRouter(true)
	original := handoffHeaderRequest("POST", "/v1/chat/completions", encodedHandoffForTest(""))

	_, first := sendHandoffHeaders(t, router, original)
	retryResponse, retry := sendHandoffHeaders(t, router, original)
	conflictResponse, conflict := sendHandoffHeaders(t, router, handoffHeaderRequest(
		"POST", "/v1/chat/completions", encodedHandoffForTest(`,"parent_invocation_id":"other-parent"`),
	))

	assert.Equal(t, handoffStatusAccepted, first.HandoffReceipt.Status)
	require.NotNil(t, retryResponse.GetRequestHeaders(), "an idempotent retry is routed normally")
	assert.Equal(t, handoffReceipt{Version: "1", ID: "handoff-test", Status: handoffStatusDuplicate, Reason: "idempotent_retry"}, retry.HandoffReceipt)
	require.NotNil(t, conflictResponse.GetImmediateResponse())
	assert.Equal(t, 409, int(conflictResponse.GetImmediateResponse().GetStatus().GetCode()))
	assert.Equal(t, handoffReceipt{Version: "1", ID: "handoff-test", Status: handoffStatusRejected, Reason: "handoff_id_conflict"}, conflict.HandoffReceipt)
}

func TestHandoffCancellationRefusesLaterRetries(t *testing.T) {
	router := handoffTransportRouter(true)
	active := handoffHeaderRequest("POST", "/v1/chat/completions", encodedHandoffForTest(""))
	cancel := handoffHeaderRequest("POST", "/v1/chat/completions", encodedHandoffForTest(`,"state":"cancelled"`))

	_, first := sendHandoffHeaders(t, router, active)
	cancelResponse, cancelled := sendHandoffHeaders(t, router, cancel)
	retryResponse, retry := sendHandoffHeaders(t, router, active)

	assert.Equal(t, handoffStatusAccepted, first.HandoffReceipt.Status)
	require.NotNil(t, cancelResponse.GetImmediateResponse(), "a cancellation is recorded, not routed")
	assert.Equal(t, 409, int(cancelResponse.GetImmediateResponse().GetStatus().GetCode()))
	assert.Equal(t, handoffReceipt{Version: "1", ID: "handoff-test", Status: handoffStatusCancelled, Reason: "cancel_recorded"}, cancelled.HandoffReceipt)
	require.NotNil(t, retryResponse.GetImmediateResponse())
	assert.Equal(t, handoffReceipt{Version: "1", ID: "handoff-test", Status: handoffStatusCancelled, Reason: "handoff_cancelled"}, retry.HandoffReceipt)
}

func TestHandoffLedgerIsSharedAcrossRouterRebuilds(t *testing.T) {
	newRouter := func() *OpenAIRouter {
		router := handoffTransportRouter(true)
		router.handoffLedger = nil
		return router
	}
	id := fmt.Sprintf("rebuild-%d", time.Now().UnixNano())
	request := handoffHeaderRequest("POST", "/v1/chat/completions", encodedHandoffWithID(id, ""))

	_, first := sendHandoffHeaders(t, newRouter(), request)
	_, afterReload := sendHandoffHeaders(t, newRouter(), request)

	assert.Equal(t, handoffStatusAccepted, first.HandoffReceipt.Status)
	assert.Equal(t, handoffStatusDuplicate, afterReload.HandoffReceipt.Status)
}

func assertHandoffHeaderNotCaptured(t *testing.T, ctx *RequestContext) {
	t.Helper()
	for key := range ctx.Headers {
		assert.False(t, strings.EqualFold(key, headers.VSRHandoffEnvelope), "raw handoff header leaked into request metadata")
	}
}
