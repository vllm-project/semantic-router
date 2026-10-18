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
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var testHandoffReceipt = handoffReceipt{
	Version: "1",
	ID:      "handoff-test",
	Status:  handoffStatusPartial,
	Reason:  handoffReasonSelectionPending,
}

func TestHandoffReceiptRidesUpstreamStreamingCacheAndSkipResponses(t *testing.T) {
	tests := []struct {
		name        string
		contentType string
		cacheHit    bool
		skip        bool
	}{
		{name: "buffered upstream", contentType: "application/json"},
		{name: "streaming upstream", contentType: "text/event-stream"},
		{name: "cache response headers", contentType: "application/json", cacheHit: true},
		{name: "skip processing response headers", contentType: "application/json", skip: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := &RequestContext{
				Headers:        map[string]string{},
				HandoffReceipt: testHandoffReceipt,
				VSRCacheHit:    test.cacheHit,
				SkipProcessing: test.skip,
			}

			response, err := (&OpenAIRouter{}).handleResponseHeaders(
				responseHeadersForHandoffTest("200", test.contentType), ctx,
			)

			require.NoError(t, err)
			mutation := response.GetResponseHeaders().GetResponse().GetHeaderMutation()
			require.NotNil(t, mutation)
			assert.Equal(t, "1", headerValueForTest(mutation, headers.VSRHandoffVersion))
			assert.Equal(t, "handoff-test", headerValueForTest(mutation, headers.VSRHandoffID))
			assert.Equal(t, handoffStatusPartial, headerValueForTest(mutation, headers.VSRHandoffStatus))
			assert.Equal(t, handoffReasonSelectionPending, headerValueForTest(mutation, headers.VSRHandoffReason))
		})
	}
}

func TestHandoffReceiptRidesImmediateResponses(t *testing.T) {
	t.Run("request header rejection", func(t *testing.T) {
		stream := NewMockStream(nil)
		ctx := &RequestContext{Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIChatV1}

		err := handoffTransportRouter(true).processRequestHeaders(
			stream, handoffHeaderRequest("POST", "/v1/chat/completions", "not+base64"), ctx,
		)

		require.NoError(t, err)
		require.Len(t, stream.Responses, 1)
		assert.Equal(t, handoffStatusRejected, immediateHeaderValue(stream.Responses[0], headers.VSRHandoffStatus))
		assert.Equal(t, "malformed_encoding", immediateHeaderValue(stream.Responses[0], headers.VSRHandoffReason))
		assert.Empty(t, immediateHeaderValue(stream.Responses[0], headers.VSRHandoffID), "no identity before parsing")
	})

	t.Run("later immediate response keeps the admission outcome", func(t *testing.T) {
		ctx := &RequestContext{HandoffReceipt: testHandoffReceipt}
		response := (&OpenAIRouter{}).createErrorResponse(400, "invalid request body")

		appendHandoffReceiptToImmediateResponse(response, ctx)

		assert.Equal(t, handoffStatusPartial, immediateHeaderValue(response, headers.VSRHandoffStatus))
		assert.Equal(t, "handoff-test", immediateHeaderValue(response, headers.VSRHandoffID))
	})

	t.Run("cache hit", func(t *testing.T) {
		ctx := exactCacheHitContext(nil)
		ctx.SourceFormat = llmprotocol.OpenAIChatV1
		ctx.TargetFormat = llmprotocol.OpenAIChatV1
		ctx.HandoffReceipt = testHandoffReceipt

		response := exactCacheHitResponse(t, &OpenAIRouter{}, ctx)
		appendHandoffReceiptToImmediateResponse(response, ctx)

		assert.Equal(t, handoffStatusPartial, immediateHeaderValue(response, headers.VSRHandoffStatus))
	})
}

func TestHandoffEnvelopeContentsNeverReachProviderReplayReceiptsOrLogs(t *testing.T) {
	const (
		opaqueRef   = "opaque-tool-state-ref-never-export"
		summary     = "confidential-task-summary"
		rootID      = "root-test"
		roleSecret  = "confidential-role"
		carrierName = headers.VSRHandoffEnvelope
	)
	encodedCarrier := encodedHandoffForTest(fmt.Sprintf(
		`,"selection":{"delegated_role":%q},"runtime":{"task_summary":%q,"tool_state_refs":[%q]}`,
		roleSecret, summary, opaqueRef,
	))
	confidential := []string{opaqueRef, summary, rootID, roleSecret, encodedCarrier}
	logCore, observedLogs := observer.New(zapcore.DebugLevel)
	restoreLogger := zap.ReplaceGlobals(zap.New(logCore))
	defer restoreLogger()

	stream := NewMockStream(nil)
	headerCtx := &RequestContext{Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIChatV1}
	require.NoError(t, handoffTransportRouter(true).processRequestHeaders(
		stream, handoffHeaderRequest("POST", "/v1/chat/completions", encodedCarrier), headerCtx,
	))
	require.Len(t, stream.Responses, 1)
	headerWire, err := protojson.Marshal(stream.Responses[0])
	require.NoError(t, err)

	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	request := testNeutralRequest(model, "ordinary user request")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.HandoffReceipt = headerCtx.HandoffReceipt
	response, err := router.handleSpecifiedModelRouting(request, model, "", ctx)
	require.NoError(t, err)
	providerMutation := response.GetRequestBody().GetResponse()
	require.NotNil(t, providerMutation)
	assert.Contains(t, providerMutation.GetHeaderMutation().GetRemoveHeaders(), carrierName)
	providerWire, err := protojson.Marshal(response)
	require.NoError(t, err)

	replayWire, err := json.Marshal(buildReplayRoutingRecord(ctx, model, model, ""))
	require.NoError(t, err)

	responseHeaders, err := router.handleResponseHeaders(responseHeadersForHandoffTest("200", "application/json"), ctx)
	require.NoError(t, err)
	receiptWire, err := protojson.Marshal(responseHeaders)
	require.NoError(t, err)

	var logText strings.Builder
	for _, entry := range observedLogs.All() {
		fmt.Fprintf(&logText, "%s %v\n", entry.Message, entry.ContextMap())
	}
	surfaces := map[string]string{
		"request header reply": string(headerWire),
		"provider request":     string(providerWire),
		"replay record":        string(replayWire),
		"receipt headers":      string(receiptWire),
		"logs":                 logText.String(),
	}
	for surface, wire := range surfaces {
		for _, value := range confidential {
			assert.NotContains(t, wire, value, "%s leaked envelope content", surface)
		}
	}
	assert.NotContains(t, string(replayWire), carrierName)
}

func responseHeadersForHandoffTest(status, contentType string) *ext_proc.ProcessingRequest_ResponseHeaders {
	return &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: status},
			{Key: "content-type", Value: contentType},
		}}},
	}
}
