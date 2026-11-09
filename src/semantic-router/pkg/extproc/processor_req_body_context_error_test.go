package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestProcessRequestBodyPreservesDecisionContextErrors(t *testing.T) {
	for _, format := range extProcMatrixFormats {
		for _, test := range []struct {
			name, code, message string
			status              int
			cause               error
		}{
			{"canceled", "request_canceled", "request canceled", 500, context.Canceled},
			{"deadline", "request_timeout", "request timed out", 504, context.DeadlineExceeded},
		} {
			t.Run(string(format)+"/"+test.name, func(t *testing.T) {
				router := routingTestRouter("worker")
				router.Config.Decisions = []config.Decision{{
					Name: "default", ModelRefs: []config.ModelRef{{Model: "worker"}},
				}}
				classifier, err := classification.NewClassifier(router.Config, nil, nil, nil)
				require.NoError(t, err)
				t.Cleanup(func() { _ = classifier.Close() })
				router.Classifier = classifier

				requestCtx, cancel := context.WithCancel(context.Background())
				cancel()
				if errors.Is(test.cause, context.DeadlineExceeded) {
					requestCtx, cancel = context.WithDeadline(context.Background(), time.Time{})
					t.Cleanup(cancel)
				}
				request := testNeutralRequest("vllm-sr/auto", "hello")
				ctx := routingTestContext(format, request)
				ctx.TraceContext = requestCtx
				encoded, err := protocolcodec.NewBuiltinEngine().EncodeRequest(format, *request, llmprotocol.Envelope{})
				require.NoError(t, err)
				stream := NewMockStream(nil)
				err = router.processRequestBody(stream, &ext_proc.ProcessingRequest_RequestBody{
					RequestBody: &ext_proc.HttpBody{Body: encoded.Body, EndOfStream: true},
				}, ctx)
				require.NoError(t, err)
				require.Len(t, stream.Responses, 1)
				response := stream.Responses[0].GetImmediateResponse()
				require.NotNil(t, response, "a failed decision must not dispatch to a backend")
				require.EqualValues(t, test.status, response.GetStatus().GetCode())
				require.NoError(t, response.ValidateAll())
				var body struct {
					Error struct {
						Code    string `json:"code"`
						Message string `json:"message"`
						Type    string `json:"type"`
					} `json:"error"`
				}
				require.NoError(t, json.Unmarshal(response.Body, &body))
				require.Equal(t, test.message, body.Error.Message)
				if format != llmprotocol.AnthropicMessagesV1 {
					require.Equal(t, test.code, body.Error.Code)
					require.Equal(t, "server_error", body.Error.Type)
				} else if test.status == 504 {
					require.Equal(t, "timeout_error", body.Error.Type)
				} else {
					require.Equal(t, "api_error", body.Error.Type)
				}
				require.ErrorIs(t, ctx.ImmediateProtocolError, test.cause)
			})
		}
	}
}

func TestSendRequestBodyErrorPreservesHTTPStatus(t *testing.T) {
	for _, format := range extProcMatrixFormats {
		for _, test := range []struct {
			input, status int
			code          string
		}{
			{415, 415, "invalid_request"},
			{504, 504, "request_timeout"},
			{499, 500, "request_failed"},
		} {
			t.Run(fmt.Sprintf("%s/%d", format, test.input), func(t *testing.T) {
				router := &OpenAIRouter{}
				ctx := routingTestContext(format, testNeutralRequest("worker", "hello"))
				stream := NewMockStream(nil)
				immediate, err := router.sendRequestBodyResult(stream, router.createErrorResponse(test.input, "boom"), nil, ctx)
				require.NoError(t, err)
				require.True(t, immediate)
				require.Len(t, stream.Responses, 1)
				response := stream.Responses[0].GetImmediateResponse()
				require.EqualValues(t, test.status, response.GetStatus().GetCode())
				require.NoError(t, response.ValidateAll())
				var body struct {
					Error struct {
						Code string `json:"code"`
					} `json:"error"`
				}
				require.NoError(t, json.Unmarshal(response.Body, &body))
				if format != llmprotocol.AnthropicMessagesV1 {
					require.Equal(t, test.code, body.Error.Code)
				}
			})
		}
	}
}
