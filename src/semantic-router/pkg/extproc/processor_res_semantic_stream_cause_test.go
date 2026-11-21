package extproc

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A model that streams tool arguments which are not JSON makes the Router end
// the stream with an error. The log has to say why, or the client's "Server
// error mid-response" cannot be traced back to the model output.
func TestSemanticStreamFailureLogsItsCause(t *testing.T) {
	core, logs := observer.New(zapcore.DebugLevel)
	defer zap.ReplaceGlobals(zap.New(core))()

	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
		RequestModel: "public", TraceContext: t.Context(),
		SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "public", Stream: true},
	}
	body := "event: message_start\ndata: " + `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\ndata: " + `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash","input":{}}}` + "\n\n" +
		"event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\": \"ls\n</ ar ar"}}` + "\n\n" +
		"event: content_block_stop\ndata: " + `{"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\ndata: " + `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}` + "\n\n" +
		"event: message_stop\ndata: " + `{"type":"message_stop"}` + "\n\n"

	(&OpenAIRouter{}).handleSemanticStreamingResponseBody([]byte(body), true, ctx)

	require.True(t, ctx.StreamingAborted)
	entries := logs.FilterMessage("neutral_stream_reconstruction_skipped").All()
	require.Len(t, entries, 1)
	require.Equal(t, "invalid_stream_tool_arguments", entries[0].ContextMap()["cause_code"])
}

// An upstream error event fails the stream too. Its message is the provider's
// own text, so the log names the failure by category and code only.
func TestSemanticStreamFailureLogsNoUpstreamMessage(t *testing.T) {
	core, logs := observer.New(zapcore.DebugLevel)
	defer zap.ReplaceGlobals(zap.New(core))()

	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
		RequestModel: "public", TraceContext: t.Context(),
		SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "public", Stream: true},
	}
	body := "event: message_start\ndata: " + `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
		"event: error\ndata: " + `{"type":"error","error":{"type":"overloaded_error","message":"upstream detail for account 42"}}` + "\n\n"

	(&OpenAIRouter{}).handleSemanticStreamingResponseBody([]byte(body), true, ctx)

	entries := logs.FilterMessage("neutral_stream_reconstruction_skipped").All()
	require.Len(t, entries, 1)
	fields := entries[0].ContextMap()
	require.NotEmpty(t, fields["error_code"])
	for key, value := range fields {
		require.NotContains(t, fmt.Sprint(value), "account 42", key)
	}
}

// A provider's error type reaches the log only when the provider documents
// it. Any other type is the provider's own text, so it logs as
// upstream_stream_error.
func TestSemanticStreamFailureLogsOnlyDocumentedUpstreamCodes(t *testing.T) {
	for errorType, want := range map[string]struct{ logged, client string }{
		"overloaded_error": {"overloaded_error", "overloaded_error"},
		"vendor_error_v2":  {"upstream_stream_error", "api_error"},
	} {
		core, logs := observer.New(zapcore.DebugLevel)
		restore := zap.ReplaceGlobals(zap.New(core))
		ctx := &RequestContext{
			SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.AnthropicMessagesV1,
			RequestModel: "public", TraceContext: t.Context(),
			SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "public", Stream: true},
		}
		body := "event: message_start\ndata: " + `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
			"event: error\ndata: " + `{"type":"error","error":{"type":"` + errorType + `","message":"upstream detail"}}` + "\n\n"

		response := (&OpenAIRouter{}).handleSemanticStreamingResponseBody([]byte(body), true, ctx)
		restore()
		// What the client reads is the encoder's, not the log's.
		require.Contains(t, string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()),
			`"type":"`+want.client+`"`, errorType)

		entries := logs.FilterMessage("neutral_stream_reconstruction_skipped").All()
		require.Len(t, entries, 1, errorType)
		require.Equal(t, want.logged, entries[0].ContextMap()["error_code"], errorType)
	}
}
