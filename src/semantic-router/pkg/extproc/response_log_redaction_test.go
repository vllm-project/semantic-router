package extproc

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

func TestSendResponseLogsOnlySafeMetadataAcrossBodyFlows(t *testing.T) {
	canaries := []string{
		"prompt-secret-canary",
		"rag-secret-canary",
		"memory-secret-canary",
		"tool-body-secret-canary",
		"bearer-auth-secret-canary",
		"cookie-secret-canary",
		"query-secret-canary",
		"token-secret-canary",
	}

	logCore, observed := observer.New(zapcore.DebugLevel)
	restoreLogger := zap.ReplaceGlobals(zap.New(logCore))
	t.Cleanup(restoreLogger)

	nonStreamingBody := []byte(`{"messages":[{"role":"user","content":"prompt-secret-canary"}],"rag":"rag-secret-canary","memory":"memory-secret-canary","tools":"tool-body-secret-canary"}`)
	nonStreamingResponse := requestBodyLogFixture(nonStreamingBody, canaries)
	nonStreamingStream := NewMockStream(nil)
	if err := sendResponse(nonStreamingStream, nonStreamingResponse, "request body"); err != nil {
		t.Fatalf("sendResponse(non-streaming) error = %v", err)
	}
	if got := nonStreamingStream.Responses[0].GetRequestBody().GetResponse().GetBodyMutation().GetBody(); string(got) != string(nonStreamingBody) {
		t.Fatal("sendResponse changed the non-streaming body")
	}

	streamingBody := []byte("data: {\"delta\":\"tool-body-secret-canary token-secret-canary\"}\n\n")
	streamingResponse := streamedResponseLogFixture(streamingBody)
	streamingStream := NewMockStream(nil)
	if err := sendResponse(streamingStream, streamingResponse, "response body"); err != nil {
		t.Fatalf("sendResponse(streaming) error = %v", err)
	}

	immediateBody := []byte(`{"error":"prompt-secret-canary query-secret-canary"}`)
	immediateResponse := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{Code: typev3.StatusCode_BadRequest},
				Body:   immediateBody,
			},
		},
	}
	if err := sendResponse(NewMockStream(nil), immediateResponse, "request body"); err != nil {
		t.Fatalf("sendResponse(immediate) error = %v", err)
	}

	errorStream := NewMockStream(nil)
	errorStream.SendError = errors.New("send-failure query-secret-canary bearer-auth-secret-canary")
	if err := sendResponse(errorStream, nonStreamingResponse, "request body"); err == nil {
		t.Fatal("sendResponse(error) returned nil")
	}

	requestHeaders := &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/chat/completions?query=" + canaries[6] + "&token=" + canaries[7]},
			{Key: "authorization", Value: "Bearer " + canaries[4]},
			{Key: "cookie", Value: "session=" + canaries[5]},
			{Key: "x-api-token", Value: canaries[7]},
		}}},
	}
	requestContext := &RequestContext{Headers: make(map[string]string)}
	_, capturedPath, _, _ := captureRequestHeaders(requestHeaders, requestContext, nil, false)
	if capturedPath != requestHeaders.RequestHeaders.Headers.Headers[1].Value {
		t.Fatal("captureRequestHeaders changed the functional request path")
	}

	replayFields := safeRouterReplayLogFields(routerreplay.RoutingRecord{
		ID:                 "replay-safe-id",
		RequestID:          "request-safe-id",
		RequestBody:        canaries[0],
		ResponseBody:       canaries[1],
		Prompt:             canaries[2],
		ToolDefinitions:    canaries[3],
		HallucinationSpans: []string{canaries[0]},
		PIIEntities:        []string{canaries[5]},
		SessionPolicy:      map[string]interface{}{"credential": canaries[7]},
		ToolTrace: &routerreplay.ToolTrace{
			Flow:      "responses",
			Stage:     "complete",
			ToolNames: []string{canaries[3]},
			Steps: []routerreplay.ToolTraceStep{{
				Type:      "tool_output",
				Arguments: canaries[3],
				Output:    canaries[1],
			}},
		},
	}, "router_replay_complete")
	logging.ComponentDebugEvent("extproc", "router_replay_complete", replayFields)

	assertObservedLogsOmitCanaries(t, observed.All(), canaries)
	assertSafeProcessingMetadata(t, observed)
	assertSafeRequestHeaderAndReplayMetadata(t, observed)
}

func TestSafeErrorForLogOmitsMessageAndPreservesTypedGRPCCode(t *testing.T) {
	const canary = "grpc-error-secret-canary"
	got := safeErrorForLog(status.Error(codes.PermissionDenied, canary))
	if strings.Contains(got, canary) {
		t.Fatalf("safeErrorForLog exposed error message: %q", got)
	}
	if !strings.Contains(got, "grpc_code=PermissionDenied") {
		t.Fatalf("safeErrorForLog result = %q, want typed gRPC status", got)
	}
}

func requestBodyLogFixture(body []byte, canaries []string) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: &ext_proc.HeaderMutation{SetHeaders: []*core.HeaderValueOption{
						{Header: &core.HeaderValue{Key: "Authorization", RawValue: []byte(canaries[4])}},
						{Header: &core.HeaderValue{Key: "Cookie", RawValue: []byte(canaries[5])}},
						{Header: &core.HeaderValue{Key: "x-api-token", RawValue: []byte(canaries[7])}},
					}},
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_Body{Body: body},
					},
				},
			},
		},
	}
}

func streamedResponseLogFixture(body []byte) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ResponseBody{
			ResponseBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_StreamedResponse{
							StreamedResponse: &ext_proc.StreamedBodyResponse{
								Body:        body,
								EndOfStream: true,
							},
						},
					},
				},
			},
		},
	}
}

func assertObservedLogsOmitCanaries(
	t *testing.T,
	entries []observer.LoggedEntry,
	canaries []string,
) {
	t.Helper()
	for _, entry := range entries {
		serialized := strings.ToLower(entry.Message + " " + fmt.Sprint(entry.ContextMap()))
		for _, canary := range canaries {
			if strings.Contains(serialized, strings.ToLower(canary)) {
				t.Fatalf("log entry %q exposed hostile canary %q: %s", entry.Message, canary, serialized)
			}
		}
	}
}

func assertSafeProcessingMetadata(t *testing.T, observed *observer.ObservedLogs) {
	t.Helper()
	ready := observed.FilterMessage("processing_response_ready").All()
	if len(ready) != 4 {
		t.Fatalf("processing_response_ready count = %d, want 4", len(ready))
	}

	nonStreamingFields := ready[0].ContextMap()
	if nonStreamingFields["response_type"] != "request_body" || nonStreamingFields["status"] != "CONTINUE" {
		t.Fatalf("unexpected non-streaming metadata: %v", nonStreamingFields)
	}
	if nonStreamingFields["set_header_count"] != int64(3) || nonStreamingFields["body_bytes"] == int64(0) {
		t.Fatalf("missing non-streaming counts: %v", nonStreamingFields)
	}

	streamingFields := ready[1].ContextMap()
	if streamingFields["body_mutation"] != "streamed_response" || streamingFields["end_of_stream"] != true {
		t.Fatalf("unexpected streaming metadata: %v", streamingFields)
	}

	immediateFields := ready[2].ContextMap()
	if immediateFields["response_type"] != "immediate_response" || immediateFields["http_status"] != "BadRequest" {
		t.Fatalf("unexpected immediate-response metadata: %v", immediateFields)
	}

	failures := observed.FilterMessage("processing_response_send_failed").All()
	if len(failures) != 1 {
		t.Fatalf("processing_response_send_failed count = %d, want 1", len(failures))
	}
	if got := fmt.Sprint(failures[0].ContextMap()["error_type"]); !strings.Contains(got, "type=") {
		t.Fatalf("error metadata = %q, want a safe type", got)
	}
}

func assertSafeRequestHeaderAndReplayMetadata(t *testing.T, observed *observer.ObservedLogs) {
	t.Helper()
	headerEntries := observed.FilterMessage("request_headers_captured").All()
	if len(headerEntries) != 1 {
		t.Fatalf("request_headers_captured count = %d, want 1", len(headerEntries))
	}
	if got := headerEntries[0].ContextMap()["path"]; got != "/v1/chat/completions" {
		t.Fatalf("logged request path = %v, want query-free path", got)
	}
	if got := headerEntries[0].ContextMap()["header_count"]; got != int64(5) {
		t.Fatalf("logged header count = %v, want 5", got)
	}

	replayEntries := observed.FilterMessage("router_replay_complete").All()
	if len(replayEntries) != 1 {
		t.Fatalf("router_replay_complete count = %d, want 1", len(replayEntries))
	}
	fields := replayEntries[0].ContextMap()
	if fields["replay_id"] != "replay-safe-id" || fields["request_id"] != "request-safe-id" {
		t.Fatalf("replay log omitted safe identifiers: %v", fields)
	}
	if fields["hallucination_span_count"] != int64(1) || fields["tool_trace_step_count"] != int64(1) {
		t.Fatalf("replay log omitted safe counts: %v", fields)
	}
}
